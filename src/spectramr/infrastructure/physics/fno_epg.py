"""Fourier Neural Operator Surrogate for Extended Phase Graph (EPG) Simulation.

Replaces the analytical EPG unrolled simulation with a learned integral operator
in the frequency domain. This provides:
- O(N log N) forward pass vs O(3^N) state evaluations for N echoes
- Full differentiability for backpropagation through complex RF waveforms
- Resolution-invariant echo spacing via spectral parameterization

Architecture:
    1. Per-voxel tissue vector [ρ, T1, T2, B1+] broadcast to 1D echo grid
    2. SpectralConv1d layers with modes = ETL//2 for echo periodicity
    3. RF/sequence parameters injected via FiLM conditioning
    4. Output: echo amplitude curve [B, 1, ETL] → effective TE echo selected

Training:
    Use ``offline_generate_dataset()`` to create ground-truth echo trains from
    the analytical EPG in ``epg.py``, then train with MSE loss. Freeze the model
    for deployment inside ``MultiPhysicsBlochLayer``.

Reference:
    Li et al., "Fourier Neural Operator for Parametric PDEs," ICLR, 2021.
    Adapted for MR Physics following Wang et al., "Neural EPG," IEEE TMI, 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """1D Fourier layer for learning echo modulation patterns.

    Operates on the 1D echo index dimension, applying learned complex-valued
    weights to low-frequency Fourier modes that capture the dominant RF
    refocusing dynamics.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """Initialize spectral convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of Fourier modes to retain (≤ N//2+1).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Xavier-like scaling for complex weights
        scale = (2 / (in_channels + out_channels)) ** 0.5
        self.weights = nn.Parameter(
            scale * torch.view_as_complex(torch.randn(in_channels, out_channels, modes, 2))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectral convolution.

        Args:
            x: Input tensor [B, C, N] where N is echo train length.

        Returns:
            Output tensor [B, out_channels, N].

        forward method for SpectralConv1d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, N = x.shape

        # 1D real FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        n_freq = x_ft.shape[-1]
        modes = min(self.modes, n_freq)

        # Truncate weights to valid modes
        w = self.weights[:, :, :modes]

        # Zero-initialized output in frequency domain
        out_ft = torch.zeros(
            B,
            self.out_channels,
            n_freq,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Apply learned spectral weights to low-frequency modes
        # einsum: (B, in, modes) * (in, out, modes) -> (B, out, modes)
        out_ft[:, :, :modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :modes], w)

        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation for RF pulse train conditioning.

    Injects sequence parameters (excitation FA, refocusing FA, echo spacing)
    into spectral features via affine transformation:
        output = γ(params) * features + β(params)
    """

    def __init__(self, param_dim: int, feature_dim: int):
        """Initialize FiLM conditioner.

        Args:
            param_dim: Dimension of conditioning parameters.
            feature_dim: Number of feature channels to modulate.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.SiLU(),
            nn.Linear(64, feature_dim * 2),  # γ and β
        )
        # Initialize near-identity: γ≈1, β≈0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            features: [B, C, N] spectral features.
            params: [B, param_dim] sequence parameters.

        Returns:
            Modulated features [B, C, N].

        forward method for FiLMConditioner.

        Executes PyTorch tensor operations.

        Args:
            features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            params (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        modulation = self.mlp(params)  # [B, 2*C]
        gamma, beta = modulation.chunk(2, dim=-1)  # Each [B, C]
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)  # [B, C, 1]
        return (1.0 + gamma) * features + beta


class FNOEPGSurrogate(nn.Module):
    """Fourier Neural Operator surrogate for EPG simulation.

    Learns the mapping from tissue parameters and RF pulse configuration
    to echo amplitudes, replacing the recursive Markovian state-tracking
    of standard EPG with a frequency-domain integral operator.

    The network consists of:
    1. Tissue lifting: maps per-voxel [ρ, T1, T2, B1+] to channel-expanded
       features broadcast along the echo index dimension
    2. FNO backbone: spectral convolutions with FiLM conditioning
    3. Projection: maps to single-channel echo amplitude curve
    """

    def __init__(
        self,
        tissue_channels: int = 4,
        hidden_width: int = 32,
        n_layers: int = 4,
        max_etl: int = 32,
        seq_param_dim: int = 3,
    ):
        """Initialize FNO EPG surrogate.

        Args:
            tissue_channels: Number of input tissue parameters (ρ, T1, T2, B1+).
            hidden_width: Width of spectral layers.
            n_layers: Number of FNO layers.
            max_etl: Maximum echo train length (determines Fourier modes).
            seq_param_dim: Sequence parameter dimension (excitation FA,
                refocusing FA, echo spacing).
        """
        super().__init__()
        self.max_etl = max_etl
        self.seq_param_dim = seq_param_dim

        # Tissue → hidden features
        self.lift = nn.Sequential(
            nn.Linear(tissue_channels, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
        )

        # Spectral layers with FiLM conditioning
        modes = max_etl // 2
        self.spectral_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv1d(hidden_width, hidden_width, modes))
            self.film_layers.append(FiLMConditioner(seq_param_dim, hidden_width))
            self.residual_convs.append(nn.Conv1d(hidden_width, hidden_width, 1))

        self.act = nn.GELU()

        # Project to echo amplitudes
        self.project = nn.Sequential(
            nn.Conv1d(hidden_width, hidden_width // 2, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_width // 2, 1, 1),
            nn.Softplus(beta=1.0),  # Echo amplitudes are non-negative
        )

    def forward(
        self,
        rho: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        b1_plus: torch.Tensor | None = None,
        te_spacing: float | torch.Tensor = 10.0,
        echo_train_length: int = 16,
        excitation_fa: float = 90.0,
        refocusing_fa: float = 180.0,
    ) -> torch.Tensor:
        """Generate echo signal from tissue parameters.

        Args:
            rho: Proton density [B, 1, H, W].
            t1: T1 relaxation time in ms [B, 1, H, W].
            t2: T2 relaxation time in ms [B, 1, H, W].
            b1_plus: B1+ transmit field map [B, 1, H, W] (optional, defaults to 1.0).
            te_spacing: Echo spacing in ms; a tensor of one value per batch row (e.g.
                ``[B, 1, 1, 1]``) gives each row its own.
            echo_train_length: Number of echoes.
            excitation_fa: Nominal excitation flip angle in degrees.
            refocusing_fa: Nominal refocusing flip angle in degrees.

        Returns:
            Signal at effective TE [B, 1, H, W].

        forward method for FNOEPGSurrogate.

        Executes PyTorch tensor operations.

        Args:
            rho (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            b1_plus (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            te_spacing (float): Expected input tensor.
            echo_train_length (int): Expected input tensor.
            excitation_fa (float): Expected input tensor.
            refocusing_fa (float): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = rho.shape
        device = rho.device

        if b1_plus is None:
            b1_plus = torch.ones_like(rho)

        # Normalize tissue parameters to [0, 1] for numerical stability. The echo amplitude
        # per unit proton density does not depend on rho, which scales the output below, so
        # its channel carries a constant: fed rho, a surrogate fitted at rho = 1 met every
        # rendered voxel out of distribution. Four channels are kept so the layer's shape
        # (and any stored weights) stay as they were.
        rho_n = torch.ones_like(rho)
        t1_n = t1.clamp(100, 5000) / 5000.0
        t2_n = t2.clamp(10, 2500) / 2500.0
        b1_n = b1_plus.clamp(0.3, 1.7) / 1.7

        # Stack tissue params: [B, 4, H, W]
        tissue = torch.cat([rho_n, t1_n, t2_n, b1_n], dim=1)

        # Flatten spatial dims: [B, 4, H*W] -> [B*H*W, 4]
        tissue_flat = tissue.view(B, 4, -1).permute(0, 2, 1).reshape(-1, 4)

        # Lift to hidden: [N_voxels, hidden_width]
        h = self.lift(tissue_flat)

        # Broadcast along echo dimension: [N_voxels, hidden_width, ETL]
        etl = echo_train_length
        h = h.unsqueeze(-1).expand(-1, -1, etl).contiguous()

        # Sequence parameter vector [N_voxels, 3], normalised for the network. B1+ scales both
        # flip angles pixel-wise; built by multiplication, not slice assignment, because the
        # B1+ map is the network's own output and autograd saves the slice being overwritten.
        flip = torch.tensor(
            [excitation_fa / 180.0, refocusing_fa / 180.0], device=device, dtype=h.dtype
        )
        b1_flat = b1_n.view(B, -1).reshape(-1, 1)  # [N_voxels, 1]
        if isinstance(te_spacing, torch.Tensor):  # one echo spacing per batch row
            te_col = te_spacing.to(device=device, dtype=h.dtype).reshape(B, 1)
            te_col = te_col.expand(B, H * W).reshape(-1, 1) / 50.0
        else:
            te_col = torch.full_like(b1_flat, te_spacing / 50.0)
        seq_params = torch.cat([flip.unsqueeze(0) * b1_flat, te_col], dim=1)

        # FNO backbone with FiLM conditioning
        for spectral, film, residual in zip(
            self.spectral_layers, self.film_layers, self.residual_convs, strict=False
        ):
            h_spec = spectral(h)
            h_spec = film(h_spec, seq_params)
            h_res = residual(h)
            h = self.act(h_spec + h_res)

        # Project to echo amplitudes: [N_voxels, 1, ETL]
        echo_curve = self.project(h)

        # Select effective TE echo (center of k-space, typically ETL//2)
        effective_echo_idx = etl // 2
        signal_flat = echo_curve[:, 0, effective_echo_idx]  # [N_voxels]

        # Reshape to spatial: [B, 1, H, W]
        signal = signal_flat.view(B, H, W).unsqueeze(1)

        # Multiply by proton density (FNO predicts normalized echo amplitude)
        signal = signal * rho

        return signal


def offline_generate_dataset(
    n_samples: int = 100_000,
    echo_train_length: int = 16,
    te_spacing: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate offline training dataset for FNO EPG surrogate.

    Samples tissue parameters uniformly from physiological ranges and computes
    ground-truth echo trains using the analytical EPG simulation.

    Args:
        n_samples: Number of tissue parameter samples.
        echo_train_length: Number of echoes per sample.
        te_spacing: Echo spacing in ms.

    Returns:
        Tuple of (tissue_params, seq_params, echo_trains):
        - tissue_params: [N, 4] tensor of (ρ, T1, T2, B1+)
        - seq_params: [N, 3] tensor of (excitation_fa, refocusing_fa, te_spacing)
        - echo_trains: [N, ETL] ground-truth echo amplitudes
    """
    from spectramr.infrastructure.physics.epg import simulate_differentiable_epg_fse

    # Sample tissue parameters from physiological ranges
    rho = torch.rand(n_samples, 1) * 0.8 + 0.2  # [0.2, 1.0]
    t1 = torch.rand(n_samples, 1) * 4900 + 100  # [100, 5000] ms
    t2 = torch.rand(n_samples, 1) * 490 + 10  # [10, 500] ms
    b1 = torch.rand(n_samples, 1) * 0.4 + 0.8  # [0.8, 1.2]

    tissue_params = torch.cat([rho, t1, t2, b1], dim=1)  # [N, 4]

    # Fixed sequence params for this batch (vary in training loop)
    excitation_fa = 90.0
    refocusing_fa = 180.0
    seq_params = (
        torch.tensor([excitation_fa, refocusing_fa, te_spacing]).unsqueeze(0).expand(n_samples, -1)
    )

    # Generate ground-truth echo trains using analytical EPG
    # Reshape to [N, 1, 1, 1] for spatial broadcasting
    rho_4d = rho.unsqueeze(-1).unsqueeze(-1)
    t1_4d = t1.unsqueeze(-1).unsqueeze(-1)
    t2_4d = t2.unsqueeze(-1).unsqueeze(-1)
    b1_4d = b1.unsqueeze(-1).unsqueeze(-1)

    echo_trains = []
    for echo_idx in range(1, echo_train_length + 1):
        # Run EPG up to each echo
        signal = simulate_differentiable_epg_fse(
            rho=rho_4d,
            t1=t1_4d,
            t2=t2_4d,
            excitation_fa=excitation_fa * b1_4d,
            refocusing_fa=refocusing_fa * b1_4d,
            echo_train_length=echo_idx,
            te_spacing=te_spacing,
        )
        echo_trains.append(signal.squeeze(-1).squeeze(-1))

    echo_trains = torch.stack(echo_trains, dim=-1).squeeze(1)  # [N, ETL]

    return tissue_params, seq_params, echo_trains
