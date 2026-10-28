"""Unit tests for ManifestLoader split subsampling (data.max_{train,val}_subjects).

The TorchIO ``SubjectsDataset`` builder is eager, so split size drives build
cost. ``_apply_subject_cap`` lets smoke/debug runs truncate each split to a
handful of subjects so the build collapses from minutes to seconds. These
tests pin the cap semantics so the smoke wrapper's ``--max-subjects`` flag
stays load-bearing.
"""

from types import SimpleNamespace

import pytest

from spectramr.data.builders.manifest_loader import ManifestLoader
from tests.utils.data_config_stub import DataConfigStub


def _records(n: int) -> list[dict]:
    return [{"primary_path": f"/tmp/subj_{i}.h5"} for i in range(n)]


class TestApplySubjectCap:
    def test_none_caps_are_noop(self):
        cfg = SimpleNamespace(split=SimpleNamespace(max_train_subjects=None, max_val_subjects=None))
        train, val = _records(10), _records(5)
        out_train, out_val = ManifestLoader._apply_subject_cap(cfg, train, val)
        assert len(out_train) == 10
        assert len(out_val) == 5

    def test_caps_truncate_each_split(self):
        cfg = SimpleNamespace(split=SimpleNamespace(max_train_subjects=4, max_val_subjects=2))
        train, val = _records(10), _records(5)
        out_train, out_val = ManifestLoader._apply_subject_cap(cfg, train, val)
        assert len(out_train) == 4
        assert len(out_val) == 2

    def test_cap_larger_than_split_keeps_all(self):
        cfg = SimpleNamespace(split=SimpleNamespace(max_train_subjects=100, max_val_subjects=100))
        train, val = _records(10), _records(5)
        out_train, out_val = ManifestLoader._apply_subject_cap(cfg, train, val)
        assert len(out_train) == 10
        assert len(out_val) == 5

    def test_cap_is_idempotent(self):
        cfg = SimpleNamespace(split=SimpleNamespace(max_train_subjects=4, max_val_subjects=2))
        train, val = _records(10), _records(5)
        once_t, once_v = ManifestLoader._apply_subject_cap(cfg, train, val)
        twice_t, twice_v = ManifestLoader._apply_subject_cap(cfg, once_t, once_v)
        assert len(twice_t) == 4
        assert len(twice_v) == 2

    def test_keeps_leading_records_in_order(self):
        cfg = SimpleNamespace(split=SimpleNamespace(max_train_subjects=3, max_val_subjects=None))
        train = _records(10)
        out_train, _ = ManifestLoader._apply_subject_cap(cfg, train, [])
        assert out_train == train[:3]

    def test_a_config_without_the_split_block_raises(self):
        """Renamed and inverted from ``test_missing_attrs_are_noop``.

        That test asserted a config lacking the cap attributes must not crash,
        which is the same defensiveness that made phase 9a's rename invisible:
        the caps are DECLARED fields on a sub-block with a ``default_factory``,
        so every real ``DataConfigSchema`` has them. A stand-in that does not is
        a stand-in of a shape nothing produces, and swallowing that silently
        retires the cap -- the smoke wrapper's ``--override
        data.split.max_train_subjects`` would stop taking effect with no symptom
        beyond a slow build.

        Kept as a raise rather than deleted: it is the one place that documents
        why the read is unguarded.
        """
        with pytest.raises(AttributeError):
            ManifestLoader._apply_subject_cap(SimpleNamespace(), _records(7), _records(3))


# ── WS1: random-split delegates to the honest split_index SSOT ─────────────────


class TestRandomSplitDelegation:
    """``load_fastmri_splits`` random-split branch must use ``split_index``:
    a single file with validation_split>0 raises (no more train==val leak),
    and a multi-file split is non-overlapping."""

    @staticmethod
    def _patch_seams(monkeypatch, full_index):
        from spectramr.data.builders import manifest_loader as ml

        monkeypatch.setattr(
            ml.ManifestLoader, "_resolve_data_root", classmethod(lambda cls, c: "/x")
        )
        monkeypatch.setattr(
            ml.ManifestLoader,
            "_extract_sensitivity_params",
            classmethod(lambda cls, c: (None, None)),
        )
        monkeypatch.setattr(
            ml.ManifestLoader,
            "_build_on_the_fly_index",
            classmethod(lambda cls, c: (list(full_index), [], [])),
        )
        monkeypatch.setattr(
            ml.ManifestLoader,
            "_apply_subject_cap",
            classmethod(lambda cls, c, tr, va: (tr, va)),
        )
        monkeypatch.setattr(
            ml.ManifestLoader,
            "_log_split_stats",
            classmethod(lambda cls, full, tr, va: None),
        )

    @staticmethod
    def _config(validation_split):
        return DataConfigStub(
            index_path=None,
            validation_index_path=None,
            holdout_site=None,
            validation_split=validation_split,
        )

    def test_single_file_random_split_raises(self, monkeypatch):
        from spectramr.data.builders.manifest_loader import ManifestLoader

        self._patch_seams(monkeypatch, [{"primary_path": "a.h5"}])
        with pytest.raises(ValueError, match="single file"):
            ManifestLoader.load_fastmri_splits(self._config(0.2))

    def test_multi_file_random_split_is_disjoint(self, monkeypatch):
        from spectramr.data.builders.manifest_loader import ManifestLoader

        full = [{"primary_path": f"{i}.h5"} for i in range(10)]
        self._patch_seams(monkeypatch, full)
        train, val = ManifestLoader.load_fastmri_splits(self._config(0.2))
        train_paths = {r["primary_path"] for r in train}
        val_paths = {r["primary_path"] for r in val}
        assert train_paths.isdisjoint(val_paths)
        assert len(val_paths) == 2 and len(train_paths) == 8

    def test_random_split_keeps_each_subject_on_one_side(self, monkeypatch):
        from spectramr.data.builders.manifest_loader import ManifestLoader

        # Two volumes per subject; a by-file split cut subject p4 in half.
        full = [{"primary_path": f"{i}.h5", "subject_id": f"p{i // 2}"} for i in range(10)]
        self._patch_seams(monkeypatch, full)
        train, val = ManifestLoader.load_fastmri_splits(self._config(0.3))
        train_subjects = {r["subject_id"] for r in train}
        val_subjects = {r["subject_id"] for r in val}
        assert train_subjects.isdisjoint(val_subjects)
        assert len(train) + len(val) == 10


class TestOnTheFlySplitRouting:
    """B19: ``_build_on_the_fly_index``'s per-source ``split`` routing.

    The plan's premise — "an unrecognised split falls into train, so a config
    typo becomes a train/val leak" — does NOT hold: ``split`` is
    ``Literal["train", "val", "both"]`` on ``DatasetSourceSchema``, so a typo is
    rejected at load and can never reach the router. What DID reach the unnamed
    ``else`` was ``both``, and these tests pin what that actually means.
    """

    @staticmethod
    def _source(name, split, path):
        return SimpleNamespace(name=name, split=split, path=str(path), data_path=None)

    @staticmethod
    def _config(sources):
        return SimpleNamespace(datasets=sources)

    @staticmethod
    def _h5_dir(tmp_path, name, n=2):
        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            (d / f"{name}_{i}.h5").write_bytes(b"")
        return d

    def test_both_lands_in_train_so_the_random_split_can_divide_it(self, tmp_path) -> None:
        """``both`` alone is CORRECT today. Its records go to ``train_index`` and
        to ``full_index``, ``val_index`` stays empty, and the caller's random
        split then draws validation from the whole pool — which is what "both"
        means. Both corpus arms using it are this shape.
        """
        src = self._source("a", "both", self._h5_dir(tmp_path, "a"))
        full, train, val = ManifestLoader._build_on_the_fly_index(self._config([src]))
        assert len(full) == 2
        assert len(train) == 2
        assert val == [], (
            "val must stay empty so the caller falls through to the random "
            "split; a non-empty val here would skip it"
        )

    def test_both_alongside_an_explicit_val_source_raises(self, tmp_path) -> None:
        """The real latent defect, and it was silent.

        The random-split fallback is skipped whenever ``val_index`` is
        non-empty. So a ``both`` source combined with an explicit ``val`` source
        contributed to TRAINING ONLY while the config said otherwise, with
        nothing downstream reporting it. 0 corpus arms today — this is a guard
        against the combination, not a fix to an affected run.
        """
        sources = [
            self._source("a", "both", self._h5_dir(tmp_path, "a")),
            self._source("b", "val", self._h5_dir(tmp_path, "b")),
        ]
        with pytest.raises(ValueError, match="split='both'"):
            ManifestLoader._build_on_the_fly_index(self._config(sources))

    def test_explicit_train_and_val_are_routed_apart(self, tmp_path) -> None:
        sources = [
            self._source("a", "train", self._h5_dir(tmp_path, "a", n=3)),
            self._source("b", "val", self._h5_dir(tmp_path, "b", n=2)),
        ]
        full, train, val = ManifestLoader._build_on_the_fly_index(self._config(sources))
        assert len(full) == 5
        assert len(train) == 3 and len(val) == 2
        assert {r["primary_path"] for r in train}.isdisjoint({r["primary_path"] for r in val})

    def test_an_unrouted_split_member_raises_rather_than_meaning_train(self, tmp_path) -> None:
        """The schema's Literal is the first line of defence; this is the second.

        If a member is later added to ``DatasetSourceSchema.split`` without a
        branch here, the old ``else`` would have silently routed held-out data
        into training. A stand-in bypasses the Literal to reach the router,
        which is the only way to exercise the guard.
        """
        src = self._source("a", "holdout", self._h5_dir(tmp_path, "a"))
        with pytest.raises(ValueError, match="does not route"):
            ManifestLoader._build_on_the_fly_index(self._config([src]))


class TestLosoEmptyValidationRaises:
    """Cohort review 2026-09-02 (T0.2): an empty LOSO validation set used to
    warn and continue while an empty TRAIN set raised two lines later, so a
    typo'd ``holdout_site`` selected every "best" checkpoint on nothing."""

    @staticmethod
    def _patch_seams(monkeypatch, full_index):
        import spectramr.data.builders.manifest_loader as ml

        monkeypatch.setattr(
            ml.ManifestLoader,
            "_extract_sensitivity_params",
            classmethod(lambda cls, c: (None, None)),
        )
        monkeypatch.setattr(
            ml.ManifestLoader,
            "_build_on_the_fly_index",
            classmethod(lambda cls, c: (list(full_index), [], [])),
        )
        monkeypatch.setattr(
            ml.ManifestLoader,
            "_apply_subject_cap",
            classmethod(lambda cls, c, tr, va: (tr, va)),
        )
        monkeypatch.setattr(
            ml.ManifestLoader,
            "_log_split_stats",
            classmethod(lambda cls, full, tr, va: None),
        )

    @staticmethod
    def _config(holdout_site):
        return DataConfigStub(
            index_path=None,
            validation_index_path=None,
            holdout_site=holdout_site,
            validation_split=0.2,
        )

    def test_holdout_matching_no_site_raises(self, monkeypatch):
        """The planted violation: no record carries the holdout site."""
        from spectramr.data.builders.manifest_loader import ManifestLoader

        full = [{"primary_path": f"{i}.h5", "metadata": {"site": "hospital_a"}} for i in range(6)]
        self._patch_seams(monkeypatch, full)
        with pytest.raises(RuntimeError, match="matched no samples"):
            ManifestLoader.load_fastmri_splits(self._config("nowhere"))

    def test_holdout_matching_a_site_partitions(self, monkeypatch):
        from spectramr.data.builders.manifest_loader import ManifestLoader

        full = [
            {
                "primary_path": f"{i}.h5",
                "metadata": {"site": "hospital_a" if i < 4 else "hospital_b"},
            }
            for i in range(6)
        ]
        self._patch_seams(monkeypatch, full)
        train, val = ManifestLoader.load_fastmri_splits(self._config("hospital_b"))
        assert len(train) == 4 and len(val) == 2
