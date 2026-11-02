"""Sampler registry — single dispatch point for diffusion samplers.

Implements ``CC-1`` from
``TODO/backlog_paradigm_expansion_roadmap.md``: a registry-dispatcher
mirroring the existing ``@register_model`` / ``@register_loss`` pattern
so new sampling paradigms (DPS / ΠGDM / DDS / EDM / RED) can be plugged
in without touching pipeline code.

The registry is the precursor to Part-I B (DPS sampler family) and is
reused by M2 (EDM) and M1 (RED).
"""

# PR-20: posterior samplers register on import.
# PR-9 (M1): Plug-and-Play / RED priors register on import.
from . import (
    annealed_langevin_dc,  # noqa: F401
    pc_sampler_consistency,  # noqa: F401
    pnp_red,  # noqa: F401
    posterior_samplers,  # noqa: F401
)
from .registry import (
    SAMPLER_REGISTRY,
    SamplerRegistry,
    get_sampler,
    list_available,
    register_sampler,
)

__all__ = [
    "SAMPLER_REGISTRY",
    "SamplerRegistry",
    "get_sampler",
    "list_available",
    "register_sampler",
]
