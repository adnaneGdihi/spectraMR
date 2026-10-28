"""Benchmarks for K-Space Operations (Phase 2)."""

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency import SimpleDataConsistency
from spectramr.infrastructure.physics.fft_ops import FFTTransformer
from spectramr.infrastructure.physics.sampling import KSpaceMaskGenerator


@pytest.fixture
def complex_image_tensors(benchmark_config):
    """Generate complex image tensors for benchmarking."""
    B, C, H, W = benchmark_config["batch_size"], 1, 256, 256
    # Create real-valued image, convert to complex representation if needed
    # FFTTransformer expects real [B, C, H, W] for forward, or complex [B, C, H, W, 2]?
    # Actually FFTTransformer.forward expects [B, C, H, W] real and returns [B, C, H, W, 2] complex
    # or [B, C, H, W] complex64.

    # Let's test with FP32 real input
    return torch.randn(
        B, C, H, W, device=benchmark_config["device"], dtype=torch.float32
    )


@pytest.fixture
def kspace_tensors(benchmark_config):
    """Generate k-space tensors for benchmarking."""
    B, C, H, W = benchmark_config["batch_size"], 1, 256, 256
    # Complex k-space
    return torch.randn(
        B, C, H, W, 2, device=benchmark_config["device"], dtype=torch.float32
    )


def benchmark_fft_forward(benchmark, complex_image_tensors, benchmark_config):
    """Benchmark FFT Forward (Image -> KSpace)."""
    fft = FFTTransformer()
    # Ensure it runs on correct device

    def _op():
        return fft.forward(complex_image_tensors)

    # Warmup
    _op()

    benchmark(_op)


def benchmark_fft_backward(benchmark, kspace_tensors, benchmark_config):
    """Benchmark FFT Backward (KSpace -> Image)."""
    fft = FFTTransformer()

    def _op():
        return fft.backward(kspace_tensors)

    # Warmup
    _op()

    benchmark(_op)


def benchmark_data_consistency(benchmark, kspace_tensors, benchmark_config):
    """Benchmark Data Consistency Projection."""
    dc = SimpleDataConsistency(method="hard")
    mask = torch.ones_like(kspace_tensors)  # Full mask

    def _op():
        return dc(kspace_tensors, kspace_tensors, mask)

    # Warmup
    _op()

    benchmark(_op)


def benchmark_mask_generation(benchmark, benchmark_config):
    """Benchmark K-Space Mask Generation."""
    mask_gen = KSpaceMaskGenerator(num_timesteps=1000, device=benchmark_config["device"])
    shape = (256, 256)

    def _op():
        return mask_gen.generate_mask(shape, acceleration=4, center_fraction=0.08)

    # Warmup
    _op()

    benchmark(_op)
