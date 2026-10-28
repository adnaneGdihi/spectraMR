"""Cache key of the coil-sensitivity estimation service."""

import hashlib

import torch

from spectramr.infrastructure.services.coil_sensitivity_service import (
    CoilSensitivityEstimationService,
)


def test_cache_key_is_a_non_security_md5(monkeypatch):
    real = hashlib.md5
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("usedforsecurity"))
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "md5", spy)
    service = object.__new__(CoilSensitivityEstimationService)
    kspace = torch.ones(2, 4, 4, dtype=torch.complex64)
    key = service._generate_cache_key(kspace, "espirit", 24, 0.02, None)
    assert seen == [False]
    assert len(key) == 32
