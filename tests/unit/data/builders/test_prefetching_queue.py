"""PrefetchingQueue and the ``subject_prefetch`` resolution (#2673).

Three claims are pinned here: the "auto" formula, that the override passes
TorchIO's subject-loader kwargs through unchanged apart from ``prefetch_factor``,
and that prefetch depth does not change which patches are drawn or their order.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchio as tio

import spectramr.data.builders.prefetching_queue as pq_mod
from spectramr.data.builders.prefetching_queue import (
    TORCH_DEFAULT_PREFETCH,
    PrefetchingQueue,
    resolve_subject_prefetch,
)


class _RandomTaggedSubjects(tio.SubjectsDataset):
    """Each loaded subject carries one torch and one NumPy draw made in the worker.

    Comparing those draws across prefetch depths checks the worker-side RNG
    streams, not only the main-process patch sampler.
    """

    def __getitem__(self, index: int) -> tio.Subject:
        subject = super().__getitem__(index)
        # Stored as constant images: the patch sampler crops images and keeps
        # construction-time attributes, but drops keys added afterwards.
        for name, value in (("torch_draw", float(torch.rand(1))), ("numpy_draw", np.random.rand())):
            subject.add_image(tio.ScalarImage(tensor=torch.full((1, 8, 8, 1), value)), name)
        return subject


def _subjects(n: int) -> list[tio.Subject]:
    return [
        tio.Subject(
            image=tio.ScalarImage(tensor=torch.full((1, 8, 8, 1), float(i))),
            sid=i,
        )
        for i in range(n)
    ]


def _shutdown(queue: tio.Queue) -> None:
    iterator = queue._subjects_iterable
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


class TestResolveSubjectPrefetch:
    def test_auto_covers_one_refill(self) -> None:
        # The issue's arm: 100 / 4 = 25 subjects per refill over 4 workers.
        assert (
            resolve_subject_prefetch("auto", queue_length=100, samples_per_volume=4, num_workers=4)
            == 7
        )

    def test_auto_rounds_partial_subjects_up(self) -> None:
        # 10 / 3 -> 4 subjects (the last one partially), over 1 worker.
        assert (
            resolve_subject_prefetch("auto", queue_length=10, samples_per_volume=3, num_workers=1)
            == 4
        )

    def test_auto_never_goes_below_torch_default(self) -> None:
        got = resolve_subject_prefetch("auto", queue_length=8, samples_per_volume=8, num_workers=4)
        assert got == TORCH_DEFAULT_PREFETCH

    def test_explicit_int_is_used_verbatim(self) -> None:
        assert (
            resolve_subject_prefetch(1, queue_length=100, samples_per_volume=4, num_workers=4) == 1
        )

    @pytest.mark.parametrize("setting", ["auto", 5])
    def test_zero_workers_resolves_to_none(self, setting: object) -> None:
        assert (
            resolve_subject_prefetch(setting, queue_length=100, samples_per_volume=4, num_workers=0)
            is None
        )

    @pytest.mark.parametrize("bad", ["AUTO", "fast", "3", True, False, 0, -1, 2.5, None])
    def test_unknown_values_raise(self, bad: object) -> None:
        with pytest.raises(ValueError, match="subject_prefetch"):
            resolve_subject_prefetch(bad, queue_length=100, samples_per_volume=4, num_workers=4)

    def test_unknown_value_raises_even_without_workers(self) -> None:
        with pytest.raises(ValueError, match="subject_prefetch"):
            resolve_subject_prefetch("fast", queue_length=100, samples_per_volume=4, num_workers=0)


class TestSubjectLoaderKwargs:
    """The override must hand torch exactly TorchIO's kwargs plus prefetch_factor."""

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch, module: object) -> list[dict]:
        calls: list[dict] = []

        def fake_loader(dataset, **kwargs):
            calls.append({"dataset": dataset, **kwargs})
            return []

        monkeypatch.setattr(module, "DataLoader", fake_loader)
        return calls

    def test_kwargs_match_torchio_except_prefetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torchio.data.queue as tio_queue_mod

        tio_calls = self._spy(monkeypatch, tio_queue_mod)
        ours_calls = self._spy(monkeypatch, pq_mod)
        dataset = tio.SubjectsDataset(_subjects(4))
        common = {
            "subjects_dataset": dataset,
            "max_length": 8,
            "samples_per_volume": 2,
            "sampler": tio.UniformSampler((4, 4, 1)),
            "num_workers": 2,
            "shuffle_subjects": True,
            "shuffle_patches": True,
        }
        plain = tio.Queue(**common)
        ours = PrefetchingQueue(**common, subject_prefetch=5)

        assert len(tio_calls) == 1 and len(ours_calls) == 1
        theirs, mine = tio_calls[0], dict(ours_calls[0])
        assert mine.pop("prefetch_factor") == 5
        assert set(mine) == set(theirs)
        for key, value in theirs.items():
            assert mine[key] is value or mine[key] == value, key
        assert plain._num_sampled_subjects == ours._num_sampled_subjects == 0

    def test_no_prefetch_kwarg_without_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch, pq_mod)
        PrefetchingQueue(
            subjects_dataset=tio.SubjectsDataset(_subjects(2)),
            max_length=4,
            samples_per_volume=2,
            sampler=tio.UniformSampler((4, 4, 1)),
            num_workers=0,
            subject_prefetch=None,
        )
        # None is torch's own default, and the only value it accepts without workers.
        assert calls[0]["prefetch_factor"] is None

    @pytest.mark.parametrize(("workers", "prefetch"), [(0, 3), (2, None)])
    def test_inconsistent_prefetch_and_workers_raise(
        self, workers: int, prefetch: int | None
    ) -> None:
        with pytest.raises(ValueError, match="subject_prefetch"):
            PrefetchingQueue(
                subjects_dataset=tio.SubjectsDataset(_subjects(2)),
                max_length=4,
                samples_per_volume=2,
                sampler=tio.UniformSampler((4, 4, 1)),
                num_workers=workers,
                subject_prefetch=prefetch,
            )


def _draw_stream(make_queue, n_patches: int) -> tuple[list[tuple], int | None]:
    torch.manual_seed(1234)
    np.random.seed(1234)
    queue = make_queue()
    try:
        stream = []
        for i in range(n_patches):
            patch = queue[i]
            stream.append(
                (
                    int(patch["sid"]),
                    tuple(int(v) for v in patch[tio.LOCATION]),
                    float(patch["torch_draw"].data[0, 0, 0, 0]),
                    float(patch["numpy_draw"].data[0, 0, 0, 0]),
                )
            )
        prefetch = getattr(queue._subjects_iterable, "_prefetch_factor", None)
        return stream, prefetch
    finally:
        _shutdown(queue)


@pytest.mark.parametrize("workers", [0, 2])
def test_prefetch_depth_does_not_change_the_patch_stream(workers: int) -> None:
    """Same seed -> same subjects, patch locations and worker draws at any prefetch.

    12 subjects x 2 patches = 24 patches per epoch; a 16-patch queue refills
    three times over the 48 draws, which cross one epoch boundary (where TorchIO
    builds a fresh subject loader).
    """
    n_subjects, spv, length, n_patches = 12, 2, 16, 48

    def build(cls, **extra):
        return lambda: cls(
            subjects_dataset=_RandomTaggedSubjects(_subjects(n_subjects)),
            max_length=length,
            samples_per_volume=spv,
            sampler=tio.UniformSampler((4, 4, 1)),
            num_workers=workers,
            shuffle_subjects=True,
            shuffle_patches=True,
            **extra,
        )

    auto = resolve_subject_prefetch(
        "auto", queue_length=length, samples_per_volume=spv, num_workers=workers
    )
    baseline, baseline_prefetch = _draw_stream(build(tio.Queue), n_patches)
    at_two, _ = _draw_stream(
        build(PrefetchingQueue, subject_prefetch=2 if workers else None), n_patches
    )
    at_auto, auto_prefetch = _draw_stream(build(PrefetchingQueue, subject_prefetch=auto), n_patches)

    assert at_two == baseline
    assert at_auto == baseline
    assert len({sid for sid, *_ in baseline}) == n_subjects
    # Guards against a vacuous pass: the worker draws must actually vary.
    assert len({draw for *_, draw in baseline}) > n_subjects
    if workers:
        # The comparison is only meaningful if the depths actually differed.
        assert (baseline_prefetch, auto_prefetch) == (2, 4)
