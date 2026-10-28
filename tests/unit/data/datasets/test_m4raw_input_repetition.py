"""``input_repetition``: which acquired M4Raw repetition a training sample serves.

Synthetic three-repetition groups on disk, CPU only, no model. Each repetition has its
own noise and its own global receiver phase, so the served input names the repetition
it came from exactly, and a target left in repetition 0's frame is ~105 deg off.
"""

from __future__ import annotations

import pickle
import random

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset, _average_reps
from spectramr.data.datasets.m4raw_identity import parse_m4raw_file_id
from spectramr.data.datasets.m4raw_input_repetition import (
    VALID_INPUT_REPETITIONS,
    InputRepetitionDraw,
    validate_input_repetition,
)

_PHASES = (0.3, 2.2, -1.5)


def _write_group(tmp_path):
    rng = np.random.default_rng(11)
    shape = (1, 2, 16, 16)
    base = np.zeros(shape, dtype=np.complex64)
    base[..., 4:12, 4:12] = 1.0
    files, reps = [], []
    for i, phase in enumerate(_PHASES, start=1):
        noisy = base + 0.05 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        img = (noisy * np.exp(1j * phase)).astype(np.complex64)
        k = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)), norm="ortho"), axes=(-2, -1)
        ).astype(np.complex64)
        path = tmp_path / f"2022_T1{i:02d}.h5"
        with h5py.File(str(path), "w") as f:
            f.create_dataset("kspace", data=k)
        files.append(path)
        reps.append(torch.from_numpy(k))
    return files, reps


def _dataset(files, **kw):
    kw.setdefault("target_mode", "phase_aligned_mean")
    return M4RawRepetitionDataset(
        h5_files=files, single_contrast=True, coil_processing_mode="none", **kw
    )


def _complex(image) -> torch.Tensor:
    """(2C, H, W, S) interleaved real/imag -> (S, C, H, W) complex."""
    t = image.data.permute(3, 0, 1, 2)
    return torch.complex(t[:, 0::2], t[:, 1::2])


def _served(subject, reps) -> int:
    inp = _complex(subject["input"])
    matches = [j for j, r in enumerate(reps) if torch.equal(inp, r)]
    assert len(matches) == 1, "the input must be exactly one acquired repetition"
    return matches[0]


def _phase_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.rad2deg(torch.angle((a * b.conj()).sum())))


def _draws(ds, reps, n: int = 24) -> list[int]:
    return [_served(ds[0], reps) for _ in range(n)]


def test_the_advertised_set() -> None:
    assert frozenset({"first", "random"}) == VALID_INPUT_REPETITIONS


def test_first_reproduces_the_legacy_input_and_target(tmp_path) -> None:
    """(a) The default is byte-identical to the loader before the knob existed."""
    files, reps = _write_group(tmp_path)
    legacy = _dataset(files)[0]
    first = _dataset(files, input_repetition="first")[0]
    for key in ("input", "target", "kspace"):
        assert torch.equal(first[key].data, legacy[key].data)
    assert torch.equal(_complex(first["input"]), reps[0])
    torch.testing.assert_close(
        _complex(first["target"]), _average_reps(reps, "phase_aligned_mean"), rtol=0, atol=0
    )
    assert first["repetition_index"] == parse_m4raw_file_id(files[0].stem).repetition


def test_random_rotates_the_input_and_aligns_the_target_to_it(tmp_path) -> None:
    """(b) Every repetition is served, each with the target in its own frame.

    PLANT: the same mean anchored to repetition 0 -- what rotating the input alone
    would serve -- sits ~105 deg from a non-zero input, so it fails the frame check.
    """
    files, reps = _write_group(tmp_path)
    ds = _dataset(files, input_repetition="random")
    torch.manual_seed(0)
    seen = set()
    for _ in range(24):
        subject = ds[0]
        j = _served(subject, reps)
        seen.add(j)
        inp, target = _complex(subject["input"]), _complex(subject["target"])
        reordered = [reps[j]] + [r for i, r in enumerate(reps) if i != j]
        torch.testing.assert_close(target, _average_reps(reordered, "phase_aligned_mean"))
        assert abs(_phase_deg(target, inp)) < 2.0
        assert subject["repetition_index"] == parse_m4raw_file_id(files[j].stem).repetition
        if j:
            rep0_frame = _average_reps(reps, "phase_aligned_mean")
            assert abs(_phase_deg(rep0_frame, inp)) > 90.0
            # Near-equal magnitude, different frame. Not exact: each repetition's
            # phase is estimated against a noisy anchor, so the two means differ slightly.
            torch.testing.assert_close(target.abs(), rep0_frame.abs(), atol=1e-3, rtol=0)
    assert seen == {0, 1, 2}


def test_random_is_reproducible_from_the_seed_and_differs_across_seeds(tmp_path) -> None:
    """A run replays from its seed; another rank/worker/epoch seed draws anew."""
    files, reps = _write_group(tmp_path)
    torch.manual_seed(5)
    ds = _dataset(files, input_repetition="random")
    first_epoch, second_epoch = _draws(ds, reps), _draws(ds, reps)
    assert first_epoch != second_epoch  # one process: the stream advances
    torch.manual_seed(5)
    assert _draws(_dataset(files, input_repetition="random"), reps) == first_epoch
    torch.manual_seed(6)
    assert _draws(_dataset(files, input_repetition="random"), reps) != first_epoch


def _repetitions(batch) -> list[int]:
    return [int(s["repetition_index"]) for s in batch]


def test_random_differs_across_workers_and_epochs_through_a_dataloader(tmp_path) -> None:
    """The production seeding: each worker gets ``base_seed + id``, and every new
    iterator draws a fresh ``base_seed`` from the parent's RNG."""
    files, _ = _write_group(tmp_path)
    ds = torch.utils.data.Subset(_dataset(files, input_repetition="random"), [0] * 12)
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2, collate_fn=_repetitions)
    epochs = [[b[0] for b in loader] for _ in range(2)]
    assert epochs[0] != epochs[1]
    # In-order delivery alternates workers: even positions are worker 0.
    assert epochs[0][0::2] != epochs[0][1::2]
    assert set(epochs[0]) | set(epochs[1]) == {1, 2, 3}


def test_random_consumes_no_global_rng(tmp_path) -> None:
    """The draw must not shift the streams masks and augmentations read."""
    files, _ = _write_group(tmp_path)

    def states_after(mode: str):
        ds = _dataset(files, input_repetition=mode)
        torch.manual_seed(3)
        np.random.seed(3)
        random.seed(3)
        for _ in range(5):
            ds[0]
        return torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()

    t_first, n_first, r_first = states_after("first")
    t_rand, n_rand, r_rand = states_after("random")
    assert torch.equal(t_first, t_rand)
    assert np.array_equal(n_first, n_rand)
    assert r_first == r_rand


def test_first_never_draws(tmp_path) -> None:
    """(c) The validation split's setting serves repetition 0 on every call."""
    files, reps = _write_group(tmp_path)
    ds = _dataset(files, input_repetition="first")
    torch.manual_seed(0)
    assert set(_draws(ds, reps)) == {0}


def test_unknown_value_raises_at_construction(tmp_path) -> None:
    """(d)"""
    files, _ = _write_group(tmp_path)
    with pytest.raises(ValueError, match="Unknown input_repetition"):
        _dataset(files, input_repetition="last")


@pytest.mark.parametrize(
    ("kw", "fragment"),
    [
        ({"target_mode": "rep_pair", "nex_target_exclude_input": True}, "target_mode='rep_pair'"),
        ({"target_mode": "complex_mean"}, "target_mode='complex_mean'"),
        ({"use_repetitions": False}, "use_repetitions=False"),
        ({"nex_target_exclude_input": True}, "leave-one-out"),
    ],
)
def test_random_raises_where_it_would_go_unread(tmp_path, kw, fragment) -> None:
    files, _ = _write_group(tmp_path)
    with pytest.raises(ValueError, match=fragment):
        _dataset(files, input_repetition="random", **kw)


def test_validator_refuses_the_cross_contrast_route_and_coil_maps() -> None:
    common = {
        "target_mode": "phase_aligned_mean",
        "use_repetitions": True,
        "nex_target_exclude_input": False,
    }
    with pytest.raises(ValueError, match="cross-contrast"):
        validate_input_repetition("random", single_contrast=False, has_coil_maps=False, **common)
    with pytest.raises(ValueError, match="coil maps"):
        validate_input_repetition("random", single_contrast=True, has_coil_maps=True, **common)
    ok = validate_input_repetition("random", single_contrast=True, has_coil_maps=False, **common)
    assert ok == "random"


def test_a_pickled_draw_reseeds_in_the_receiving_process() -> None:
    """A worker never inherits the parent's stream position."""
    draw = InputRepetitionDraw("random")
    torch.manual_seed(1)
    draw(3)
    clone = pickle.loads(pickle.dumps(draw))
    assert clone.mode == "random" and clone._rng is None
    assert draw(1) == 0  # one loaded repetition: nothing to choose
    with pytest.raises(ValueError, match="no repetitions"):
        draw(0)
