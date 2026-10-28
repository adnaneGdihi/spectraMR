.. _module_loader:

Asynchronous Data Loading
=========================

**Where this lives:** ``src/spectramr/infrastructure/builders/directors/data_pipeline_director.py``
(``DataPipelineDirector``, which resolves the policy) and
``src/spectramr/infrastructure/builders/leaf/data_builders.py`` (``DataLoaderBuilder``,
which constructs the loader). The data-SSOT rule confines loader construction to
these builders — a higher layer never calls ``DataLoader(...)`` itself.

High-Level Logic
----------------
Loading is asynchronous so that CPU-bound preparation (disk I/O, decompression,
TorchIO transforms) overlaps GPU-bound model execution, in a consumer-producer
arrangement. spectraMR does not implement its own loader class: it configures
PyTorch's worker-process prefetch pipeline and owns the *policy* for how that
pipeline is sized.

Operational mechanisms:

1.  **Worker processes.** ``data.loader.num_workers`` (default ``4``) moves
    preparation into separate processes. The director is the single choke point for
    every training loader it builds, and the declared count is a **ceiling**: it is
    clamped down to this rank's share of the node's cores and never raised, so an
    arm that already fits is untouched. Without that term a 4-rank node spawned 4×
    the declared decoders, which is how a multi-GPU run could come out slower than
    a single-GPU one. Train, validation and inference loaders are clamped
    separately (``role="train"`` / ``"val"`` / ``"inference"``).
2.  **Pinned memory.** ``data.loader.pin_memory`` (default ``True``) allocates
    page-locked staging buffers for faster host-to-device DMA. It is ANDed with
    accelerator availability, because pinning is a no-op and a warning on a
    CPU-only run, and an explicit ``pin_memory: false`` is honoured rather than
    overwritten by a hardcoded ``True``.
3.  **Prefetch depth.** ``data.loader.prefetch_factor`` (default ``2``) keeps
    ``prefetch_factor × num_workers`` samples in flight across all workers, so the
    GPU does not wait on Python overhead. ``persistent_workers`` (default
    ``True``) keeps worker processes alive between epochs, trading resident memory
    for the per-epoch respawn cost. It is forced off per loader where persistence
    cannot help or torch rejects it: a loader whose clamped worker count is ``0``
    (the validation default), and the single-pass loaders built by the inference,
    ``make`` and ``data_utils`` paths. The TorchIO patch-queue path is unaffected,
    because ``PrefetchingQueue`` does not pass persistence to its subject loader.
    The value each loader actually received is recorded as ``persistent`` beside
    its worker decision in the run provenance.
4.  **Reserved cores.** ``data.loader.reserved_cores`` (default ``2``) withholds
    cores per rank for the main training process and its helper threads, so the
    ceiling on workers is ``max(1, cpus_per_rank - reserved_cores)``. It lowers
    the ceiling only; it never raises ``num_workers``. Example: with
    ``--cpus-per-task=4`` and ``num_workers: 4`` the run gets 2 workers; request
    ``--cpus-per-task = num_workers + 2`` (6 here) to keep all 4. Set
    ``reserved_cores: 0`` to restore the previous behaviour.

Configuration
-------------

.. code-block:: yaml

   data:
     loader:
       num_workers: 4          # a ceiling; clamped to this rank's core share
       pin_memory: true        # ANDed with accelerator availability
       prefetch_factor: 2      # 2 x num_workers samples in flight
       persistent_workers: true   # forced off for 0-worker and single-pass loaders
       reserved_cores: 2          # ceiling = max(1, cpus_per_rank - reserved_cores)

Sizing note: workers are the dominant data-side memory term — the default of 4
adds roughly 1.3 GB over ``num_workers: 0``, and 8 adds about 2.5 GB. Size against
the node, not against the number of cores it reports.

Class Breakdown
---------------

.. autoclass:: spectramr.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector
   :members:

.. autoclass:: spectramr.infrastructure.builders.leaf.data_builders.DataLoaderBuilder
   :members:
   :undoc-members:

.. autoclass:: spectramr.config.schemas.data.DataLoaderConfigSchema
   :members:
