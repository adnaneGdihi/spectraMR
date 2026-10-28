"""A ``tio.Queue`` whose subject loader prefetches enough subjects for one refill.

``tio.Queue._fill`` runs in the main process whenever the patch list is empty and
pulls ``ceil(max_length / samples_per_volume)`` subjects from an internal
``DataLoader`` that TorchIO 1.2.1 builds with torch's default ``prefetch_factor=2``.
Only ``2 * num_workers`` subjects are therefore in flight when a refill starts,
and the GPU waits for the rest (#2673). :class:`PrefetchingQueue` is the one
owner of the subject loader's ``prefetch_factor``; :func:`resolve_subject_prefetch`
is the one owner of the ``data.sampling.subject_prefetch`` knob's meaning.

Prefetch depth changes how many subjects are in flight, not which subjects are
drawn or in what order: the subject loader is map-style with ``in_order=True``,
so index ``k`` is always handled by worker ``k mod num_workers`` and each worker
consumes its indices FIFO. ``tests/unit/data/builders/test_prefetching_queue.py``
pins this by comparing patch streams at prefetch 2 and at the resolved value.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torchio as tio
from torch.utils.data import DataLoader

from spectramr.config.schemas.data import SUBJECT_PREFETCH_AUTO, validate_subject_prefetch

__all__ = [
    "SUBJECT_PREFETCH_AUTO",
    "TORCH_DEFAULT_PREFETCH",
    "PrefetchingQueue",
    "resolve_subject_prefetch",
]

# torch's DataLoader default, and what every queue received before #2673. The
# "auto" resolution never goes below it, so no run gets less prefetch than before.
TORCH_DEFAULT_PREFETCH = 2


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def resolve_subject_prefetch(
    setting: object,
    *,
    queue_length: int,
    samples_per_volume: int,
    num_workers: int,
) -> int | None:
    """Resolve the subject loader's ``prefetch_factor`` for one queue.

    Args:
        setting: ``"auto"`` or an explicit int >= 1.
        queue_length: the queue's ``max_length``.
        samples_per_volume: patches drawn per subject.
        num_workers: the subject loader's worker count *after* any topology clamp.

    Returns:
        ``None`` when ``num_workers == 0`` (torch rejects a ``prefetch_factor``
        without workers, and the main process loads subjects synchronously
        anyway); otherwise the per-worker prefetch depth. ``"auto"`` resolves to
        ``ceil(ceil(queue_length / samples_per_volume) / num_workers)``, floored
        at :data:`TORCH_DEFAULT_PREFETCH`, so one refill's subjects are already
        loading while the previous refill's patches train.

    Raises:
        ValueError: ``setting`` is neither ``"auto"`` nor an int >= 1.
    """
    checked = validate_subject_prefetch(setting)
    if num_workers <= 0:
        return None
    if checked == SUBJECT_PREFETCH_AUTO:
        subjects_per_refill = _ceil_div(queue_length, samples_per_volume)
        return max(TORCH_DEFAULT_PREFETCH, _ceil_div(subjects_per_refill, num_workers))
    return int(checked)


class PrefetchingQueue(tio.Queue):  # type: ignore[misc,unused-ignore]  # torchio is untyped
    """``tio.Queue`` whose internal subject loader takes an explicit ``prefetch_factor``.

    ``persistent_workers`` is deliberately not passed: ``tio.Queue`` builds a new
    subject ``DataLoader`` each time the previous iterator is exhausted, so
    persistent workers would belong to a discarded loader and stay alive until
    garbage collection without ever being reused.

    No ``worker_init_fn`` is passed either, matching TorchIO: torch already seeds
    ``random``, ``numpy`` and ``torch`` in each worker, and adding
    ``spectramr.core.worker_seeding.seed_worker`` here would reseed NumPy with a
    different value and change the augmentation draws of every existing run.
    """

    def __init__(self, *args: Any, subject_prefetch: int | None, **kwargs: Any) -> None:
        # Set before super().__init__: with start_background=True TorchIO builds
        # the subject loader inside its constructor, through the override below.
        self.subject_prefetch = subject_prefetch
        super().__init__(*args, **kwargs)

    def _get_subjects_iterable(self) -> Iterator[tio.Subject]:
        # Mirrors torchio 1.2.1 Queue._get_subjects_iterable; only prefetch_factor
        # is added. test_prefetching_queue pins the kwargs against TorchIO's own.
        if (self.subject_prefetch is None) != (self.num_workers <= 0):
            raise ValueError(
                f"subject_prefetch={self.subject_prefetch} with num_workers={self.num_workers}: "
                "resolve it with resolve_subject_prefetch, which returns an int exactly "
                "when there are workers to prefetch."
            )
        subjects_loader = DataLoader(
            self.subjects_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            collate_fn=self._get_first_item,
            sampler=self.subject_sampler,
            shuffle=self.shuffle_subjects,
            prefetch_factor=self.subject_prefetch,
        )
        self._num_sampled_subjects = 0
        return iter(subjects_loader)
