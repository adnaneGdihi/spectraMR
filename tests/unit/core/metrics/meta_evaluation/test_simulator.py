"""Tests for the meta-evaluation simulator's real-asset threading.

Covers the WS1 wiring: ``run_simulator`` attaches per-content acquisition assets
to each ``DegradationSample``, and ``_build_sample_context`` prefers those real
assets over the synthetic birdcage stand-in — falling back to synthesis only for
fields the acquisition did not supply.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.meta_evaluation.simulator import (
    SimulatorConfig,
    _build_sample_context,
    run_simulator,
)


def _noise(clean: torch.Tensor, theta: float, gen: torch.Generator) -> torch.Tensor:
    return clean + theta * torch.randn(clean.shape, generator=gen)


def _config() -> SimulatorConfig:
    return SimulatorConfig(
        families=["gaussian_noise"],
        n_severities_per_family=2,
        operator_library={"gaussian_noise": _noise},
    )


def test_run_simulator_attaches_assets_by_content():
    clean = torch.zeros(1, 1, 8, 8)
    ctx = MetricContext(coil_maps=torch.ones(1, 4, 8, 8), acceleration=1.0)
    samples = run_simulator(
        [("subj0", clean), ("subj1", clean)],
        _config(),
        assets_by_content={"subj0": ctx},
    )
    by_content = {s.content_id: s for s in samples}
    # subj0 samples carry the real assets; subj1 (absent from the map) get None.
    assert all(s.assets is ctx for s in samples if s.content_id == "subj0")
    assert all(s.assets is None for s in samples if s.content_id == "subj1")
    assert by_content["subj0"].assets is ctx


def test_run_simulator_without_assets_is_backward_compatible():
    clean = torch.zeros(1, 1, 8, 8)
    samples = run_simulator([("subj0", clean)], _config())
    assert samples and all(s.assets is None for s in samples)


def test_build_sample_context_prefers_real_coil_maps():
    clean = torch.rand(1, 1, 16, 16)
    real_smaps = torch.ones(1, 4, 16, 16, dtype=torch.complex64)
    real = MetricContext(coil_maps=real_smaps, acceleration=3.0)

    ctx = _build_sample_context(
        clean, ("coil_maps", "y_kspace", "acceleration"), real=real
    )
    # Real coil maps used verbatim (identity), NOT the synthetic birdcage.
    assert ctx.coil_maps is real_smaps
    assert ctx.acceleration == 3.0
    # y_kspace was still synthesised (real assets did not supply it).
    assert ctx.y_kspace is not None


def test_build_sample_context_synthesises_when_no_real_assets():
    clean = torch.rand(1, 1, 16, 16)
    ctx = _build_sample_context(clean, ("coil_maps", "foreground_mask"), real=None)
    assert ctx.coil_maps is not None  # synthetic birdcage stand-in
    assert ctx.foreground_mask is not None


def test_precompute_passes_context_as_keyword_so_nr_metric_computes():
    """Raw registry metrics declare ``*, context=None`` (keyword-only).

    ``precompute_metric_values`` must pass the context as a KEYWORD — a
    positional third arg TypeErrors and is swallowed into NaN, which had been
    silently NaN-ing the entire no-reference battery on the raw-metric path.
    Regression: a context-needing metric (``ndcr``) must return finite values.
    """
    import math

    from spectramr.core.metrics import get_metric
    from spectramr.core.metrics.meta_evaluation.simulator import precompute_metric_values
    from spectramr.infrastructure.physics.asset_preparation import prepare_metric_context

    c, h, w = 4, 16, 16
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    phantom = torch.exp(-(yy**2 + xx**2) / 0.3).clamp(0, 1)[None, None].float()
    smaps = torch.ones(1, c, h, w, dtype=torch.complex64) / (c**0.5)
    asset = prepare_metric_context(
        coil_images=phantom.to(torch.complex64) * smaps, smaps=smaps, p99=1.0
    )

    samples = run_simulator(
        [("subj_0", phantom.squeeze(0))],
        _config(),
        assets_by_content={"subj_0": asset},
    )
    vals = precompute_metric_values(samples, {"ndcr": get_metric("ndcr")})["ndcr"]
    assert vals and all(math.isfinite(v) for v in vals)


# The ids that collided under the old 8-byte prefix hash (phantom ids share bytes 5-8,
# M4Raw tags share the scan month, fastMRI ids share the ``file_bra`` prefix).
_COLLIDING_IDS = (
    "phantom_0",
    "phantom_1",
    "phantom_12",
    "2022091411_T1",
    "2022091411_FLAIR",
)


def test_every_content_gets_its_own_seed_and_its_own_draw():
    clean = torch.zeros(1, 1, 8, 8)
    samples = run_simulator([(cid, clean) for cid in _COLLIDING_IDS], _config())
    first = {s.content_id: s for s in samples if s.severity["theta"] == samples[0].severity["theta"]}
    seeds = {cid: first[cid].seed for cid in _COLLIDING_IDS}
    assert len(set(seeds.values())) == len(_COLLIDING_IDS), seeds
    outs = [first[cid].degraded for cid in _COLLIDING_IDS]
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            assert not torch.equal(outs[i], outs[j]), (_COLLIDING_IDS[i], _COLLIDING_IDS[j])


def test_content_seed_is_the_blake2b_v1_scheme():
    import hashlib

    from spectramr.core.metrics.meta_evaluation.simulator import SEED_SCHEME, content_seed

    expected = (
        int.from_bytes(hashlib.blake2b(b"7:phantom_1", digest_size=8).digest(), "big")
        & 0xFFFFFFFF
    )
    assert SEED_SCHEME == "blake2b_v1"
    assert content_seed(7, "phantom_1") == expected


def test_subject_of_is_attached_to_every_sample():
    clean = torch.zeros(1, 1, 8, 8)
    samples = run_simulator(
        [("s1#0", clean), ("s1#1", clean)], _config(), subject_of={"s1#0": "s1", "s1#1": "s1"}
    )
    assert {s.subject_id for s in samples} == {"s1"}


def test_subject_of_must_cover_every_content_and_nothing_else():
    clean = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="no subject"):
        run_simulator([("a", clean), ("b", clean)], _config(), subject_of={"a": "s"})
    with pytest.raises(ValueError, match="never simulated"):
        run_simulator([("a", clean)], _config(), subject_of={"a": "s", "z": "s"})
    with pytest.raises(ValueError, match="empty subject"):
        run_simulator([("a", clean)], _config(), subject_of={"a": ""})


def test_subject_id_is_keyword_only():
    import dataclasses

    from spectramr.core.metrics.meta_evaluation.types import DegradationSample

    fields = {f.name: f for f in dataclasses.fields(DegradationSample)}
    assert fields["subject_id"].kw_only is True


def _lib_config(**kw) -> SimulatorConfig:
    return SimulatorConfig(
        families=["gaussian_noise"],
        n_severities_per_family=3,
        operator_library={"gaussian_noise": _noise},
        **kw,
    )


def test_samples_carry_physical_and_damage():
    clean = torch.rand(1, 8, 8)
    cfg = _lib_config(
        severity_annotators={"gaussian_noise": lambda t: {"physical": 10 * t, "native": t}},
        damage_yardsticks=("rel_l2", "grad_l2"),
    )
    samples = run_simulator([("s0", clean)], cfg)
    from spectramr.core.metrics.meta_evaluation.damage import damage

    for s in samples:
        th = s.severity["theta"]
        assert s.severity["physical"] == 10 * th and s.severity["native"] == th
        for y in ("rel_l2", "grad_l2"):
            assert s.severity[f"damage_{y}"] == pytest.approx(damage(clean, s.degraded, y), rel=1e-5)


def test_no_axes_no_yardsticks_severity_is_exactly_theta():
    samples = run_simulator([("s0", torch.rand(1, 8, 8))], _lib_config())
    assert all(set(s.severity) == {"theta"} for s in samples)


def test_unknown_yardstick_raises():
    with pytest.raises(ValueError, match="nope"):
        _lib_config(damage_yardsticks=("nope",))


def test_annotators_must_cover_every_family():
    with pytest.raises(ValueError, match="gaussian_noise"):
        _lib_config(severity_annotators={})
