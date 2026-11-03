"""Physics Operator Implementations

This package contains concrete implementations of forward operators
for MRI reconstruction that are registered with the physics registry.
"""

# Import and register operators
from .fft_operator import FFT2DOperator
from .nufft_operator import NUFFTOperator

__all__ = [
    "FFT2DOperator",
    "NUFFTOperator",
]
