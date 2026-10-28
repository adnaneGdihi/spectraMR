"""Regression: the multi-step cold-diffusion VALIDATION path must process the
batch in ``validation.val_chunk_size`` micro-batches.

Root cause (experiment_11_kspace_cold_diffusion, 2026-06-14): when
``validation.multistep_cold_sampling`` is on, ``_generate_validation_prediction``
ran ``gen.sample()`` over the *whole* validation batch. The single-step path has
a ``val_chunk_size`` micro-chunking guard (``_forward_chunked``) precisely to cap
peak GPU memory, but the multi-step reverse-sampling path bypassed it. The
iterative reverse pass holds a full fp32 forward's activations of the heavy
nested complex / dual-domain UNet, so a batch of 2 peaked ~40 GiB and OOM'd the
39.95 GiB (0.9x 44 GiB) per-process cap. Every one of the run's 40 validations
failed (``validation_metrics.csv`` all-``1.0`` sentinel) and the early-stopping
monitor never appeared.

The reverse sampler reconstructs each batch element independently
(``PhysicsInformedColdDiffusion.sample`` uses a per-element timestep vector and no
cross-batch reduction), so chunking along dim 0 and concatenating is numerically
identical to one full-batch call. These tests pin that behaviour on the extracted
``_sample_multistep_chunked`` helper, called unbound on a stub (mirroring
``test_diffusion_validation_sampler_steps_ws1``) to avoid building the heavy
strategy.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.utils.config_block_stub import block_stub

import torch

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


def _strategy(val_chunk_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            validation=block_stub(
            "validation",
            val_chunk_size=val_chunk_size)
        )
    )


def _recording_gen() -> tuple[SimpleNamespace, list[dict]]:
    """A stub generator whose ``sample`` echoes the measurement and records the
    per-call batch sizes and forwarded step count."""
    calls: list[dict] = []

    def sample(
        measurement: torch.Tensor,
        mask: torch.Tensor | None,
        inference_timesteps: int | None,
        smaps: torch.Tensor | None = None,
        start_timestep: int | None = None,
    ) -> torch.Tensor:
        calls.append(
            {
                "batch": int(measurement.shape[0]),
                "mask_batch": None if mask is None else int(mask.shape[0]),
                "steps": inference_timesteps,
                "smaps": smaps,
            }
        )
        # The real sampler returns k-space of the same batch; echo a marked
        # tensor so the concatenation order can be verified.
        return measurement + 1.0

    return SimpleNamespace(sample=sample), calls


def test_chunks_batch_by_val_chunk_size() -> None:
    """val_chunk_size=1 must split a batch of 2 into two single-sample calls."""
    gen, calls = _recording_gen()
    strat = _strategy(val_chunk_size=1)
    meas = torch.arange(2 * 8 * 4 * 4, dtype=torch.float32).reshape(2, 8, 4, 4)
    mask = torch.ones(2, 1, 4, 4)

    out = DiffusionTrainingStrategy._sample_multistep_chunked(
        strat, gen, meas, mask, steps=10
    )

    assert [c["batch"] for c in calls] == [1, 1]
    assert [c["mask_batch"] for c in calls] == [1, 1]
    assert all(c["steps"] == 10 for c in calls)
    assert out.shape == meas.shape


def test_no_split_when_batch_within_chunk() -> None:
    """A batch that fits the chunk runs as a single full-batch call (no overhead,
    identical to the pre-fix behaviour for small batches)."""
    gen, calls = _recording_gen()
    strat = _strategy(val_chunk_size=2)
    meas = torch.zeros(2, 8, 4, 4)
    mask = torch.ones(2, 1, 4, 4)

    out = DiffusionTrainingStrategy._sample_multistep_chunked(
        strat, gen, meas, mask, steps=5
    )

    assert [c["batch"] for c in calls] == [2]
    assert out.shape == meas.shape


def test_chunked_result_is_numerically_identical_to_full_batch() -> None:
    """Per-sample reconstruction is independent, so chunked+concatenated output
    must equal the full-batch output element-for-element (preserving order)."""
    meas = torch.randn(4, 8, 4, 4)
    mask = torch.ones(4, 1, 4, 4)

    gen_full, _ = _recording_gen()
    full = DiffusionTrainingStrategy._sample_multistep_chunked(
        _strategy(val_chunk_size=4), gen_full, meas, mask, steps=7
    )

    gen_chunked, calls = _recording_gen()
    chunked = DiffusionTrainingStrategy._sample_multistep_chunked(
        _strategy(val_chunk_size=1), gen_chunked, meas, mask, steps=7
    )

    assert [c["batch"] for c in calls] == [1, 1, 1, 1]
    torch.testing.assert_close(full, chunked)


def test_handles_none_mask() -> None:
    """The mask may be None (sampler infers it); each chunk must still receive
    None rather than a sliced tensor."""
    gen, calls = _recording_gen()
    strat = _strategy(val_chunk_size=1)
    meas = torch.zeros(2, 8, 4, 4)

    out = DiffusionTrainingStrategy._sample_multistep_chunked(
        strat, gen, meas, None, steps=3
    )

    assert [c["batch"] for c in calls] == [1, 1]
    assert [c["mask_batch"] for c in calls] == [None, None]
    assert out.shape == meas.shape


def test_each_chunk_receives_its_own_samples_maps() -> None:
    """The maps are per-subject: a whole-batch tensor handed to a chunk would
    project one subject's k-space through another's coils (#1424, #1914)."""
    gen, calls = _recording_gen()
    strat = _strategy(val_chunk_size=1)
    meas = torch.zeros(2, 8, 4, 4)
    smaps = torch.arange(2, dtype=torch.float32).view(2, 1, 1, 1).expand(2, 4, 4, 4)

    DiffusionTrainingStrategy._sample_multistep_chunked(
        strat, gen, meas, torch.ones(2, 1, 4, 4), steps=10, smaps=smaps
    )
    assert [c["smaps"].shape[0] for c in calls] == [1, 1]
    assert [float(c["smaps"][0, 0, 0, 0]) for c in calls] == [0.0, 1.0]
