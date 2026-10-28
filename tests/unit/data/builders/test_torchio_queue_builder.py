"""Unit tests for TorchIO Queue Builder (Phase T2).

Tests the TorchIOQueueConfig dataclass and TorchIOQueueBuilder
for correct queue and loader construction.
"""

from unittest.mock import patch as mock_patch

import pytest
import torch
import torchio as tio

from spectramr.data.builders.prefetching_queue import PrefetchingQueue
from spectramr.data.builders.torchio_queue_builder import (
    TorchIOQueueBuilder,
    TorchIOQueueConfig,
)
from tests.utils.data_config_stub import DataConfigStub


class TestTorchIOQueueConfig:
    """Tests for TorchIOQueueConfig dataclass."""

    def test_default_construction(self):
        """Test that default config is valid."""
        config = TorchIOQueueConfig()
        assert config.queue_length == 256
        assert config.samples_per_volume == 16
        assert config.patch_size == (256, 256)
        assert config.batch_size == 8
        assert config.num_workers == 0
        assert config.use_queue_for_validation is False

    def test_custom_values(self):
        """Test with custom values."""
        config = TorchIOQueueConfig(
            queue_length=512,
            samples_per_volume=32,
            patch_size=(512, 512),
            batch_size=16,
            num_workers=4,
        )
        assert config.queue_length == 512
        assert config.samples_per_volume == 32
        assert config.patch_size == (512, 512)
        assert config.batch_size == 16
        assert config.num_workers == 4

    def test_invalid_queue_length(self):
        """Test that invalid queue_length raises error."""
        with pytest.raises(ValueError, match="queue_length"):
            TorchIOQueueConfig(queue_length=0)

        with pytest.raises(ValueError, match="queue_length"):
            TorchIOQueueConfig(queue_length=-1)

    def test_invalid_samples_per_volume(self):
        """Test that invalid samples_per_volume raises error."""
        with pytest.raises(ValueError, match="samples_per_volume"):
            TorchIOQueueConfig(samples_per_volume=0)

    def test_invalid_batch_size(self):
        """Test that invalid batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size"):
            TorchIOQueueConfig(batch_size=0)

    def test_invalid_num_workers(self):
        """Test that invalid num_workers raises error."""
        with pytest.raises(ValueError, match="num_workers"):
            TorchIOQueueConfig(num_workers=-1)

    def test_invalid_patch_size(self):
        """Test that invalid patch_size raises error."""
        with pytest.raises(ValueError, match="patch_size"):
            TorchIOQueueConfig(patch_size=())

        with pytest.raises(ValueError, match="patch_size"):
            TorchIOQueueConfig(patch_size=(0, 256))

    def test_shuffling_options(self):
        """Test shuffling configuration options."""
        config = TorchIOQueueConfig(
            shuffle_subjects_train=False,
            shuffle_patches_train=False,
            shuffle_subjects_val=True,
            shuffle_patches_val=True,
        )
        assert config.shuffle_subjects_train is False
        assert config.shuffle_patches_train is False
        assert config.shuffle_subjects_val is True
        assert config.shuffle_patches_val is True

    def test_from_training_config_minimal(self):
        """Test from_training_config with minimal config."""

        minimal_config = lambda: DataConfigStub(
                queue_length=256,
                samples_per_volume=16,
                patch_size=(256, 256),
                batch_size=8,
                num_workers=0,
                pin_memory=False,
                max_prefetch=2,
                persistent_workers=False,
            )
        config = TorchIOQueueConfig.from_training_config(minimal_config())
        assert config.queue_length == 256
        assert config.samples_per_volume == 16

    def test_from_training_config_full(self):
        """Test from_training_config with full config."""

        full_config = lambda: DataConfigStub(
                queue_length=512,
                samples_per_volume=32,
                patch_size=(512, 512),
                batch_size=16,
                num_workers=4,
                pin_memory=False,
                max_prefetch=2,
                persistent_workers=False,
                use_queue_for_validation=True,
            )
        config = TorchIOQueueConfig.from_training_config(full_config())
        assert config.queue_length == 512
        assert config.samples_per_volume == 32
        assert config.patch_size == (512, 512)
        assert config.batch_size == 16
        assert config.num_workers == 4
        assert config.use_queue_for_validation is True

    def test_from_training_config_legacy_slice_aware(self):
        """Test legacy slice_aware parameter mapping."""

        legacy_config = lambda: DataConfigStub(
                slice_aware=True,
                queue_length=256,
                samples_per_volume=16,
                patch_size=(256, 256),
                batch_size=8,
                num_workers=0,
                pin_memory=False,
                max_prefetch=2,
                persistent_workers=False,
            )
        config = TorchIOQueueConfig.from_training_config(legacy_config())
        assert config.use_queue_for_validation is True


class TestTorchIOQueueBuilder:
    """Tests for TorchIOQueueBuilder class."""

    def create_dummy_dataset(self, num_volumes: int = 2) -> tio.SubjectsDataset:
        """Create a dummy dataset for testing."""
        subjects = []
        for i in range(num_volumes):
            # Create a random 3D image
            data = torch.randn(1, 64, 64, 64)
            image = tio.ScalarImage(tensor=data)
            subject = tio.Subject(image=image)
            subjects.append(subject)

        return tio.SubjectsDataset(subjects)

    def test_build_train_queue_returns_tuple(self):
        """Test that build_train_queue returns (queue, loader) tuple."""
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(patch_size=(32, 32, 32))

        queue, loader = TorchIOQueueBuilder.build_train_queue(dataset, config)

        assert isinstance(queue, tio.Queue)
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_build_val_queue_queue_based(self):
        """Test validation queue with queue-based strategy."""
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(
            patch_size=(32, 32, 32),
            use_queue_for_validation=True,
        )

        queue, loader = TorchIOQueueBuilder.build_val_queue(dataset, config)

        assert isinstance(queue, tio.Queue)
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_build_val_queue_direct(self):
        """Test validation queue with direct strategy."""
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(
            patch_size=(32, 32, 32),
            use_queue_for_validation=False,
        )

        queue, loader = TorchIOQueueBuilder.build_val_queue(dataset, config)

        assert queue is None  # No queue for direct strategy
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_direct_val_loader_seeds_workers(self):
        """Reliability regression (2026-07-02): the direct-val loader has real
        workers but carried no ``worker_init_fn``, so numpy-based transforms
        could duplicate draws across its workers. It must now seed workers via
        the canonical ``seed_worker`` hook (post-WS5: re-exported from
        ``core.worker_seeding`` by this builder module)."""
        from spectramr.data.builders.torchio_queue_builder import seed_worker

        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(
            patch_size=(32, 32, 32),
            use_queue_for_validation=False,
            num_workers=2,
        )
        _queue, loader = TorchIOQueueBuilder.build_val_queue(dataset, config)
        assert loader.worker_init_fn is seed_worker

    def test_queue_length_configuration(self):
        """Test that queue length is set correctly."""
        dataset = self.create_dummy_dataset()

        for queue_length in [64, 128, 256, 512]:
            config = TorchIOQueueConfig(
                queue_length=queue_length,
                patch_size=(32, 32, 32),
            )

            queue, _ = TorchIOQueueBuilder.build_train_queue(dataset, config)

            # Queue's max_length attribute
            assert queue.max_length == queue_length

    def test_batch_size_configuration(self):
        """Test that batch size is set correctly."""
        dataset = self.create_dummy_dataset()

        for batch_size in [4, 8, 16]:
            config = TorchIOQueueConfig(
                batch_size=batch_size,
                patch_size=(32, 32, 32),
            )

            _, loader = TorchIOQueueBuilder.build_train_queue(dataset, config)

            assert loader.batch_size == batch_size

    def test_train_loader_drops_last_partial_batch(self):
        """Train loader must drop the last incomplete batch.

        Patch sampling yields ``samples_per_volume`` patches per volume, which
        is rarely a multiple of ``batch_size``; a size-1 trailing batch breaks
        BatchNorm and skews running stats. Training must drop it.
        """
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(batch_size=8, patch_size=(32, 32, 32))

        _, loader = TorchIOQueueBuilder.build_train_queue(dataset, config)

        assert loader.drop_last is True

    def test_shuffling_train(self):
        """Test shuffling configuration for training."""
        dataset = self.create_dummy_dataset()

        # With shuffling
        config_shuffle = TorchIOQueueConfig(
            shuffle_subjects_train=True,
            shuffle_patches_train=True,
            patch_size=(32, 32, 32),
        )
        queue_shuffle, _ = TorchIOQueueBuilder.build_train_queue(
            dataset, config_shuffle
        )
        assert queue_shuffle.shuffle_subjects is True
        assert queue_shuffle.shuffle_patches is True

        # Without shuffling
        config_no_shuffle = TorchIOQueueConfig(
            shuffle_subjects_train=False,
            shuffle_patches_train=False,
            patch_size=(32, 32, 32),
        )
        queue_no_shuffle, _ = TorchIOQueueBuilder.build_train_queue(
            dataset, config_no_shuffle
        )
        assert queue_no_shuffle.shuffle_subjects is False
        assert queue_no_shuffle.shuffle_patches is False

    def test_val_queue_reduced_length(self):
        """Test that validation queue uses reduced length."""
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(
            queue_length=256,
            patch_size=(32, 32, 32),
            use_queue_for_validation=True,
        )

        queue, _ = TorchIOQueueBuilder.build_val_queue(dataset, config)

        # Validation queue should use queue_length // 2
        assert queue.max_length == 128

    def test_val_queue_no_shuffle(self):
        """Test that validation queue has no shuffling."""
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(
            patch_size=(32, 32, 32),
            use_queue_for_validation=True,
        )

        queue, _ = TorchIOQueueBuilder.build_val_queue(dataset, config)

        # Validation should not shuffle
        assert queue.shuffle_subjects is False
        assert queue.shuffle_patches is False

    def test_get_queue_stats(self):
        """Test queue statistics computation."""
        config = TorchIOQueueConfig(
            queue_length=256,
            samples_per_volume=16,
            patch_size=(256, 256),
            batch_size=8,
            num_workers=4,
        )

        stats = TorchIOQueueBuilder.get_queue_stats(config)

        assert stats["queue_length"] == 256
        assert stats["samples_per_volume"] == 16
        assert stats["batch_size"] == 8
        assert stats["num_workers"] == 4
        assert stats["patch_elements"] == 256 * 256
        assert "estimated_memory_mb" in stats
        assert stats["estimated_memory_mb"] > 0

    def test_queue_stats_memory_calculation(self):
        """Test memory calculation in queue stats."""
        config = TorchIOQueueConfig(
            queue_length=512,
            patch_size=(256, 256),
        )

        stats = TorchIOQueueBuilder.get_queue_stats(config)

        # 256*256 elements * 4 bytes/element * 2 channels * 512 patches / (1024^2)
        expected_mb = (256 * 256 * 4 * 2 * 512) / (1024**2)
        assert abs(stats["estimated_memory_mb"] - expected_mb) < 1.0

    def test_direct_validation_batch_size_one(self):
        """Test that direct validation uses batch_size=1."""
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(
            batch_size=16,
            use_queue_for_validation=False,
        )

        _, loader = TorchIOQueueBuilder.build_val_queue(dataset, config)

        # Direct validation uses batch_size=1 regardless of config
        assert loader.batch_size == 1


class TestQueueBuilderIntegration:
    """Integration tests for queue builder."""

    def create_dummy_dataset(self, num_volumes: int = 4) -> tio.SubjectsDataset:
        """Create a dummy dataset for testing."""
        subjects = []
        for i in range(num_volumes):
            data = torch.randn(1, 64, 64, 64)
            image = tio.ScalarImage(tensor=data)
            subject = tio.Subject(image=image)
            subjects.append(subject)
        return tio.SubjectsDataset(subjects)

    def test_complete_train_pipeline(self):
        """Test building a complete training pipeline."""
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(
            queue_length=256,
            samples_per_volume=16,
            patch_size=(32, 32, 32),
            batch_size=8,
        )

        queue, loader = TorchIOQueueBuilder.build_train_queue(dataset, config)

        assert isinstance(queue, tio.Queue)
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert len(loader) > 0

    def test_complete_validation_pipeline_queue(self):
        """Test building a complete validation pipeline (queue-based)."""
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(
            queue_length=128,
            patch_size=(32, 32, 32),
            batch_size=8,
            use_queue_for_validation=True,
        )

        queue, loader = TorchIOQueueBuilder.build_val_queue(dataset, config)

        assert isinstance(queue, tio.Queue)
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert len(loader) > 0

    def test_complete_validation_pipeline_direct(self):
        """Test building a complete validation pipeline (direct)."""
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig(
            queue_length=128,
            patch_size=(32, 32, 32),
            use_queue_for_validation=False,
        )

        queue, loader = TorchIOQueueBuilder.build_val_queue(dataset, config)

        assert queue is None
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert len(loader) == len(dataset)  # One volume per batch

    def test_from_config_to_queue_pipeline(self):
        """Test full pipeline from config object to queue."""

        training_config = lambda: DataConfigStub(
                queue_length=256,
                samples_per_volume=16,
                patch_size=(32, 32, 32),
                batch_size=8,
                num_workers=0,
                pin_memory=False,
                max_prefetch=2,
                persistent_workers=False,
                use_queue_for_validation=False,
            )
        dataset = self.create_dummy_dataset()
        config = TorchIOQueueConfig.from_training_config(training_config())

        train_queue, train_loader = TorchIOQueueBuilder.build_train_queue(
            dataset, config
        )
        val_queue, val_loader = TorchIOQueueBuilder.build_val_queue(dataset, config)

        assert isinstance(train_queue, tio.Queue)
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert val_queue is None
        assert isinstance(val_loader, torch.utils.data.DataLoader)


class TestTorchIOQueueBuilderStaticContract:
    """Regression tests for static-vs-instance signature drift (BTQ-001/002)."""

    def test_build_train_queue_callable_as_staticmethod(self):
        """BTQ-001: build_train_queue must be a staticmethod.

        Mirrors the build_val_queue contract: the underlying descriptor in
        the class __dict__ is a ``staticmethod`` whose ``__func__`` is the
        plain function, so unbound calls and instance calls behave the same.
        """
        descriptor = TorchIOQueueBuilder.__dict__["build_train_queue"]
        assert isinstance(descriptor, staticmethod)
        assert callable(descriptor.__func__)
        # The attribute access form yields the underlying function directly.
        assert TorchIOQueueBuilder.build_train_queue is descriptor.__func__
        # Symmetric with the sibling val builder, which has always been static.
        val_descriptor = TorchIOQueueBuilder.__dict__["build_val_queue"]
        assert isinstance(val_descriptor, staticmethod)

    def test_filter_subjects_decorator_is_single_staticmethod(self):
        """BTQ-002: exactly one layer of @staticmethod wrapping.

        A double ``@staticmethod`` would produce ``staticmethod(staticmethod(fn))``,
        breaking ``type(...) is staticmethod`` introspection.
        """
        descriptor = TorchIOQueueBuilder.__dict__["_filter_patch_compatible_subjects"]
        assert type(descriptor) is staticmethod
        # One unwrap reaches the plain function (not another staticmethod).
        assert not isinstance(descriptor.__func__, staticmethod)
        assert callable(descriptor.__func__)


class TestFromTrainingConfigQueueLengthRejection:
    """``queue_length <= 0`` must RAISE, never auto-correct (2026-06 audit).

    Supersedes BTQ-003, which asserted the silent auto-correct's log message.
    The auto-correct guarded the 7 ``experiment_11*`` arms that set
    ``queue_length: 0`` — all migrated since (no YAML in the repo sets a
    non-positive value, and both schema paths now enforce ``ge=1``). A
    silently rewritten queue length is pitfall #9/#10: the run looks green
    while sampling from a queue the user never configured.
    """

    @staticmethod
    def _zero_queue_config():
        # queue_length=0 must reach the builder as an authored value, so it is
        # set on the stub rather than left to the schema default of 200.
        cfg = DataConfigStub(
            samples_per_volume=16,
            patch_size=(32, 32, 32),
            batch_size=8,
            num_workers=0,
            pin_memory=False,
            max_prefetch=2,
            persistent_workers=False,
        )
        cfg.sampling = cfg.sampling.model_copy(update={"queue_length": 0})
        return cfg

    def test_from_training_config_queue_length_zero_raises(self):
        with pytest.raises(ValueError, match="queue_length"):
            TorchIOQueueConfig.from_training_config(self._zero_queue_config())

    def test_from_training_config_no_silent_autocorrect_warning(self):
        """No warning-path remains: the failure is loud, not logged-and-continued."""
        with mock_patch(
            "spectramr.data.builders.torchio_queue_builder.logger.warning"
        ) as mock_warning:
            with pytest.raises(ValueError):
                TorchIOQueueConfig.from_training_config(self._zero_queue_config())
        for call in mock_warning.call_args_list:
            assert "auto-corrected" not in str(call)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestDirectValidationNoPatchFilter:
    """Direct (full-volume) validation must NOT drop patch-incompatible subjects.

    The F10/E35 patch-compat pre-filter exists for *patch sampling* (the tio.Queue
    paths): a subject smaller than patch_size would crash the sampler. Direct
    validation (``use_queue_for_validation=False``) loads full volumes — no
    sampler — so the same subjects are perfectly valid there. Applying the
    filter unconditionally dropped them (and crashed with 'Subjects list is
    empty' when ALL validation volumes were smaller than the train patch_size,
    e.g. 64³ M4Raw volumes under the default 256² patch).
    """

    @staticmethod
    def _small_subjects_dataset(n: int = 3) -> tio.SubjectsDataset:
        subjects = [
            tio.Subject(image=tio.ScalarImage(tensor=torch.randn(1, 64, 64, 64)))
            for _ in range(n)
        ]
        return tio.SubjectsDataset(subjects)

    def test_direct_val_keeps_subjects_smaller_than_patch(self):
        dataset = self._small_subjects_dataset(3)
        config = TorchIOQueueConfig(
            batch_size=4,
            patch_size=(256, 256),  # larger than every subject
            use_queue_for_validation=False,
        )

        queue, loader = TorchIOQueueBuilder.build_val_queue(dataset, config)

        assert queue is None  # direct strategy
        assert len(loader.dataset) == 3  # nothing filtered

    def test_queue_val_still_filters_incompatible_subjects(self):
        """The queue-based val path KEEPS the filter (sampler would crash)."""
        dataset = self._small_subjects_dataset(3)
        config = TorchIOQueueConfig(
            batch_size=4,
            patch_size=(256, 256),
            use_queue_for_validation=True,
        )

        with pytest.raises(ValueError, match="[Ss]ubjects"):
            TorchIOQueueBuilder.build_val_queue(dataset, config)


def test_build_train_queue_raises_on_full_sampler() -> None:
    """``sampler.type='full'`` (whole-volume) must NOT silently fall back to a
    UniformSampler patch queue on the training path (CLAUDE.md #9).

    Full-slice training is routed through the director's no-Queue DataLoaderBuilder
    bypass; if 'full' (or the inference-only 'grid') reaches ``build_train_queue``
    that is a wiring bug, so it must raise — not degrade to random 128² patches,
    which is exactly the silent fallback that hid the ULF 'patches of a slice' bug.
    """
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 8, 8, 1)))
    ds = tio.SubjectsDataset([subj])
    config = TorchIOQueueConfig(patch_size=(4, 4, 1))
    config.sampler_type = "full"
    with pytest.raises(ValueError, match="must not reach the training"):
        TorchIOQueueBuilder.build_train_queue(ds, config)


def test_require_dry_iter_fails_loud_for_dataset_without_it() -> None:
    """A queued dataset lacking dry_iter must fail at BUILD, not mid-training.

    ``tio.Queue.iterations_per_epoch`` calls ``subjects_dataset.dry_iter()``; the
    pre-flight guard converts the cryptic first-step AttributeError (oracle_bssfp
    / exp_vf_29, 2026-06) into a clear build-time error (CLAUDE.md #9 fail-loud).
    """
    import pytest

    from spectramr.data.builders.torchio_queue_builder import _require_dry_iter

    class _NoDryIter:
        def __len__(self) -> int:
            return 1

    with pytest.raises(TypeError, match="dry_iter"):
        _require_dry_iter(_NoDryIter(), "train")

    class _HasDryIter(_NoDryIter):
        def dry_iter(self) -> list:
            return []

    _require_dry_iter(_HasDryIter(), "validation")  # callable dry_iter → no raise


# ── WS4: prefetch comes from the authoritative prefetch_factor knob ────────────


class TestPrefetchFromConfig:
    """``from_training_config`` must read ``prefetch_factor`` (the authoritative
    knob), and honor a legacy config that only set ``max_prefetch`` (folded into
    ``prefetch_factor`` at schema-load)."""

    def test_reads_prefetch_factor(self):
        from spectramr.config.schemas.data import DataConfigSchema

        qc = TorchIOQueueConfig.from_training_config(
            DataConfigSchema(prefetch_factor=7)
        )
        assert qc.prefetch_factor == 7

    def test_legacy_max_prefetch_is_honored_via_fold(self):
        from spectramr.config.schemas.data import DataConfigSchema

        # max_prefetch alone → folded into prefetch_factor by the schema.
        qc = TorchIOQueueConfig.from_training_config(
            DataConfigSchema(max_prefetch=5)
        )
        assert qc.prefetch_factor == 5


# ── B11: an epoch that yields ZERO batches must fail at BUILD time ────────────


class TestTrainQueueZeroBatchGuard:
    """A training epoch that produces no batch at all must raise, not run green.

    The train loader sets ``drop_last=True`` (pinned by
    ``test_train_loader_drops_last_partial_batch``), so one epoch over the queue
    yields ``floor(n_subjects * samples_per_volume / batch_size)`` batches. When
    that product is smaller than ``batch_size`` the loader yields NOTHING: the
    training loop iterates zero times, takes zero optimizer steps, and the run
    still reports SUCCESS. That is pitfall #10 without even a warning to look
    at -- the checkpoint is the random initialisation and every metric grades it.
    """

    @staticmethod
    def _dataset(num_volumes: int) -> tio.SubjectsDataset:
        return tio.SubjectsDataset(
            [
                tio.Subject(image=tio.ScalarImage(tensor=torch.randn(1, 64, 64, 64)))
                for _ in range(num_volumes)
            ]
        )

    def test_zero_batch_epoch_raises_naming_both_counts(self):
        config = TorchIOQueueConfig(
            patch_size=(32, 32, 32),
            samples_per_volume=2,
            batch_size=8,  # 1 subject x 2 patches = 2 < 8 → zero batches
        )

        with pytest.raises(ValueError) as excinfo:
            TorchIOQueueBuilder.build_train_queue(self._dataset(1), config)

        message = str(excinfo.value)
        # Asserted by substring, not ``match=``: that argument is ``re.search``
        # and the message carries literal ``patch(es)`` / ``subject(s)``, whose
        # parentheses would be read as regex groups.
        assert "ZERO batches" in message
        assert "= 2 patch" in message  # the computed patch count
        assert "batch_size=8" in message  # what one step actually consumes

    def test_exactly_one_full_batch_does_not_raise(self):
        """The boundary is ``<``, not ``<=`` -- exactly one batch is a valid epoch."""
        config = TorchIOQueueConfig(
            patch_size=(32, 32, 32),
            samples_per_volume=4,
            batch_size=8,  # 2 subjects x 4 patches == 8 == batch_size
        )

        queue, loader = TorchIOQueueBuilder.build_train_queue(self._dataset(2), config)

        assert isinstance(queue, tio.Queue)
        assert len(loader) == 1  # the one full batch survives drop_last

    def test_dataset_with_an_unusable_len_skips_the_guard(self):
        """A dataset that cannot report its length is skipped, not crashed.

        The patch count is unknowable for such a dataset, so the guard steps
        aside rather than inventing a second failure mode. The ``except
        TypeError`` must actually run, though: the stub counts the call and the
        test asserts it fired, because a bare "did not crash" assertion would
        pass just as well with the whole guard deleted.

        ``shuffle_subjects_train`` is off deliberately. ``tio.Queue`` builds an
        internal ``DataLoader(shuffle=self.shuffle_subjects)``, and a shuffling
        loader constructs a ``RandomSampler`` that calls ``len`` in its own
        ``__init__`` -- the TypeError would then escape from inside TorchIO,
        before the guard is ever reached. Do not "tidy" this back to the default.
        """

        class _NoUsableLen:
            """Delegates to a real SubjectsDataset but refuses to be measured."""

            def __init__(self, inner: tio.SubjectsDataset) -> None:
                self._inner = inner
                self.len_calls = 0

            def __getitem__(self, index):
                return self._inner[index]  # also drives iteration (raises IndexError)

            def __len__(self) -> int:
                self.len_calls += 1
                raise TypeError("length is not knowable for a streaming dataset")

            def dry_iter(self):
                return self._inner.dry_iter()

        dataset = _NoUsableLen(self._dataset(1))
        config = TorchIOQueueConfig(
            patch_size=(32, 32, 32),
            samples_per_volume=1,
            batch_size=64,  # would raise loudly if the length were knowable
            shuffle_subjects_train=False,
        )

        queue, loader = TorchIOQueueBuilder.build_train_queue(dataset, config)

        assert isinstance(queue, tio.Queue)
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert dataset.len_calls >= 1, (
            "the guard never consulted len() -- this test would pass with the "
            "zero-batch guard removed entirely"
        )


# ── #2673: the train queue's subject loader prefetches one refill ─────────────


class TestTrainQueueSubjectPrefetch:
    """``build_train_queue`` must return a queue whose live subject loader carries
    the resolved ``prefetch_factor``. The planted case swaps in the plain
    ``tio.Queue`` (torch's default of 2) and shows the same check turns red."""

    @staticmethod
    def _dataset() -> tio.SubjectsDataset:
        return tio.SubjectsDataset(
            [tio.Subject(image=tio.ScalarImage(tensor=torch.rand(1, 8, 8, 1))) for _ in range(6)]
        )

    @staticmethod
    def _config(**overrides) -> TorchIOQueueConfig:
        # 16 / 2 = 8 subjects per refill over 2 workers -> auto resolves to 4.
        base = {
            "queue_length": 16,
            "samples_per_volume": 2,
            "patch_size": (4, 4, 1),
            "batch_size": 2,
            "num_workers": 2,
        }
        return TorchIOQueueConfig(**{**base, **overrides})

    @staticmethod
    def _assert_subject_prefetch(queue: tio.Queue, expected: int) -> None:
        iterator = queue._subjects_iterable
        try:
            assert isinstance(queue, PrefetchingQueue), type(queue).__name__
            assert iterator._prefetch_factor == expected
        finally:
            iterator._shutdown_workers()

    def test_train_queue_subject_loader_gets_resolved_prefetch(self):
        queue, _ = TorchIOQueueBuilder.build_train_queue(self._dataset(), self._config())
        assert queue.subject_prefetch == 4
        self._assert_subject_prefetch(queue, 4)

    def test_explicit_subject_prefetch_reaches_the_loader(self):
        queue, _ = TorchIOQueueBuilder.build_train_queue(
            self._dataset(), self._config(subject_prefetch=3)
        )
        self._assert_subject_prefetch(queue, 3)

    def test_zero_workers_passes_no_prefetch(self):
        queue, _ = TorchIOQueueBuilder.build_train_queue(
            self._dataset(), self._config(num_workers=0)
        )
        assert isinstance(queue, PrefetchingQueue)
        assert queue.subject_prefetch is None
        assert queue._subjects_iterable._num_workers == 0

    def test_planted_plain_tio_queue_fails_the_check(self, monkeypatch):
        import spectramr.data.builders.torchio_queue_builder as builder_mod

        def plain_queue(*args, subject_prefetch, **kwargs):
            return tio.Queue(*args, **kwargs)

        monkeypatch.setattr(builder_mod, "PrefetchingQueue", plain_queue)
        queue, _ = TorchIOQueueBuilder.build_train_queue(self._dataset(), self._config())
        assert queue._subjects_iterable._prefetch_factor == 2
        with pytest.raises(AssertionError):
            self._assert_subject_prefetch(queue, 4)

    def test_resolution_is_logged(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="spectramr.data.builders.torchio_queue_builder"):
            queue, _ = TorchIOQueueBuilder.build_train_queue(self._dataset(), self._config())
        queue._subjects_iterable._shutdown_workers()
        assert "declared='auto', resolved=4 per worker x 2 worker(s)" in caplog.text

    @pytest.mark.parametrize("bad", ["fast", 0, True])
    def test_config_rejects_unknown_subject_prefetch(self, bad):
        with pytest.raises(ValueError, match="subject_prefetch"):
            TorchIOQueueConfig(subject_prefetch=bad)

    def test_from_training_config_threads_subject_prefetch(self):
        from spectramr.config.schemas.data import DataConfigSchema

        default = TorchIOQueueConfig.from_training_config(DataConfigSchema())
        explicit = TorchIOQueueConfig.from_training_config(
            DataConfigSchema(sampling={"subject_prefetch": 6})
        )
        assert default.subject_prefetch == "auto"
        assert explicit.subject_prefetch == 6
