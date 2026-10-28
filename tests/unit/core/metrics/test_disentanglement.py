"""Tests for the disentanglement readings: linear probes, swap distances, the accumulator.

Each reading is shown to turn red on a *planted* violation: a content code that carries the
contrast label must score a high leak, and one that does not must sit at chance.
"""

from __future__ import annotations

import collections

import pytest
import torch

from spectramr.core.metrics.disentanglement import (
    DisentanglementAccumulator,
    cosine_distance,
    linear_probe,
    relative_error,
)


def _blocked_labels(blocks: int = 4, width: int = 6, n_classes: int = 3) -> torch.Tensor:
    """Contiguous runs per label, cycled so each half of the rows holds every label."""
    return torch.arange(n_classes).repeat_interleave(width).repeat(blocks)


def _noise(n: int, d: int = 8, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _label_code(labels: torch.Tensor, d: int = 8, seed: int = 1) -> torch.Tensor:
    """A code that encodes the label: a per-class direction plus noise."""
    directions = torch.randn(
        int(labels.max()) + 1, d, generator=torch.Generator().manual_seed(seed)
    )
    return 3.0 * directions[labels] + 0.1 * _noise(labels.numel(), d, seed + 1)


def _volumes_slice_major(subjects: int = 12, slices: int = 6):
    """Rows ordered slice-by-slice across every volume, features = a per-volume fingerprint.

    The fingerprint carries no contrast information (it is random per volume), but with many
    slices per volume a probe that sees a volume in training recognises it in testing and
    reads its label back. Contiguous halves put every volume on both sides here.
    """
    gen = torch.Generator().manual_seed(11)
    n_vol = subjects * 3
    fingerprint = torch.randn(n_vol, 8, generator=gen)
    vol_label = torch.arange(n_vol) % 3
    vol_subject = [f"S{v // 3:02d}" for v in range(n_vol)]
    rows = [(v, s) for s in range(slices) for v in range(n_vol)]
    feats = torch.stack([fingerprint[v] + 0.01 * torch.randn(8, generator=gen) for v, _ in rows])
    labels = torch.tensor([int(vol_label[v]) for v, _ in rows])
    groups = [vol_subject[v] for v, _ in rows]
    return feats, labels, groups


class TestSubjectDisjointSplit:
    def test_contiguous_halves_read_a_volume_fingerprint_as_a_leak(self) -> None:
        feats, labels, _ = _volumes_slice_major()
        accuracy, chance = linear_probe(feats, labels)
        assert accuracy - chance > 0.2  # measured 0.30; the artefact the grouped split removes

    def test_a_split_by_subject_sees_no_leak_in_a_fingerprint(self) -> None:
        feats, labels, groups = _volumes_slice_major()
        accuracy, chance = linear_probe(feats, labels, groups=groups)
        assert accuracy - chance < 0.25

    def test_a_label_carried_by_the_code_is_still_found_across_subjects(self) -> None:
        _, labels, groups = _volumes_slice_major()
        accuracy, _ = linear_probe(_label_code(labels), labels, groups=groups)
        assert accuracy > 0.9

    def test_one_subject_cannot_be_split(self) -> None:
        labels = _blocked_labels()
        with pytest.raises(ValueError, match="at least 2 subjects"):
            linear_probe(_noise(labels.numel()), labels, groups=["S00"] * labels.numel())

    def test_group_and_row_counts_must_agree(self) -> None:
        labels = _blocked_labels()
        with pytest.raises(ValueError, match="group labels"):
            linear_probe(_noise(labels.numel()), labels, groups=["a", "b"])

    def test_the_accumulator_requires_groups_on_every_batch_or_none(self) -> None:
        acc = DisentanglementAccumulator()
        acc.add(torch.zeros(2, 4, 1, 1), torch.zeros(2, 4), torch.zeros(2), groups=["a", "a"])
        with pytest.raises(ValueError, match="every batch"):
            acc.add(torch.zeros(2, 4, 1, 1), torch.zeros(2, 4), torch.zeros(2))

    def test_the_accumulator_reports_which_split_it_used(self) -> None:
        feats, labels, groups = _volumes_slice_major()
        acc = DisentanglementAccumulator()
        acc.add(feats.view(-1, 8, 1, 1), feats, labels, groups=groups)
        assert acc.probes()["val_probe_split_by_subject"] == 1.0
        acc2 = DisentanglementAccumulator()
        acc2.add(feats.view(-1, 8, 1, 1), feats, labels)
        assert acc2.probes()["val_probe_split_by_subject"] == 0.0


class TestLinearProbe:
    def test_a_code_that_carries_the_label_is_decoded(self) -> None:
        labels = _blocked_labels()
        accuracy, chance = linear_probe(_label_code(labels), labels)
        assert accuracy > 0.95
        assert chance < 0.5

    def test_a_code_without_the_label_sits_at_chance(self) -> None:
        labels = _blocked_labels(blocks=40)
        accuracy, chance = linear_probe(_noise(labels.numel(), seed=7), labels)
        assert accuracy - chance < 0.15  # planted-clean control: no leak to find

    def test_chance_is_the_constant_predictor_not_one_over_k(self) -> None:
        labels = torch.tensor([0, 1] * 4 + [0, 0, 0, 0, 0, 1, 0, 0])  # test half is 7/8 label 0
        _, chance = linear_probe(_noise(labels.numel()), labels)
        assert chance == pytest.approx(7 / 8)

    @pytest.mark.parametrize(
        ("labels", "match"),
        [
            (torch.zeros(12, dtype=torch.long), "at least 2 contrast labels"),
            (torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]), "training side holds one"),
            (torch.tensor([0, 1, 0]), "at least 4 rows"),
        ],
    )
    def test_degenerate_validation_orders_raise_instead_of_reporting(
        self, labels: torch.Tensor, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            linear_probe(_noise(labels.numel()), labels)

    def test_row_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="feature rows"):
            linear_probe(_noise(10), _blocked_labels())


class TestCosineDistance:
    def test_identical_orthogonal_and_opposite(self) -> None:
        a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-6)
        assert cosine_distance(a, a[:, [1, 0]]) == pytest.approx(1.0, abs=1e-6)
        assert cosine_distance(a, -a) == pytest.approx(2.0, abs=1e-6)

    def test_rows_are_flattened(self) -> None:
        x = torch.randn(3, 2, 4, 4)
        assert cosine_distance(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal shapes"):
            cosine_distance(torch.ones(2, 3), torch.ones(2, 4))


class TestRelativeError:
    def test_zero_when_the_code_came_back_and_one_when_it_was_lost(self) -> None:
        ref = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        assert relative_error(ref, ref) == pytest.approx(0.0, abs=1e-7)
        assert relative_error(torch.zeros_like(ref), ref) == pytest.approx(1.0, abs=1e-6)

    def test_a_one_dimensional_code_still_registers_a_wrong_scale(self) -> None:
        """A cosine of 1-D codes is a sign; this sees 0.1 against 0.5."""
        assert relative_error(torch.tensor([[0.1]]), torch.tensor([[0.5]])) == pytest.approx(0.8)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal shapes"):
            relative_error(torch.ones(2, 3), torch.ones(2, 4))


class TestAccumulator:
    def _fill(self, content_leaks: bool) -> DisentanglementAccumulator:
        labels = _blocked_labels(blocks=6)
        style = _label_code(labels, seed=3)
        content = _label_code(labels, seed=5) if content_leaks else _noise(labels.numel(), seed=9)
        acc = DisentanglementAccumulator()
        for lo in range(0, labels.numel(), 6):
            sl = slice(lo, lo + 6)
            acc.add(content[sl].view(-1, 8, 1, 1), style[sl], labels[sl])
        return acc

    def test_factorised_codes_read_as_style_high_and_content_leak_low(self) -> None:
        out = self._fill(content_leaks=False).probes()
        assert out["val_style_contrast_acc"] > 0.9
        assert out["val_content_contrast_leak"] < 0.3

    def test_planted_content_leak_is_reported(self) -> None:
        out = self._fill(content_leaks=True).probes()
        assert out["val_content_contrast_leak"] > 0.4

    def test_an_empty_pass_reports_nothing_rather_than_a_number(self) -> None:
        assert DisentanglementAccumulator().probes() == {}

    def test_batch_row_counts_must_agree(self) -> None:
        with pytest.raises(ValueError, match="content rows"):
            DisentanglementAccumulator().add(
                torch.zeros(2, 4, 1, 1), torch.zeros(3, 4), torch.zeros(2)
            )

    def test_swap_rows_are_capped_per_label_and_pair_across_labels_only(self) -> None:
        acc = DisentanglementAccumulator(per_label_swap_rows=2)
        for label in (0, 0, 0, 1, 1, 2):
            if acc.wants(label):
                acc.keep(label, (label, f"row{label}"))
        pairs = acc.swap_pairs(max_pairs=100)
        assert all(a[0] != b[0] for a, b in pairs)
        assert len(pairs) == 2 * 2 * 2 + 2 * 1 * 2 + 2 * 1 * 2  # ordered pairs across 3 labels
        assert len(acc.swap_pairs(max_pairs=3)) == 3
        assert not acc.wants(0)

    def test_without_style_only_the_content_reading_is_reported(self) -> None:
        """Where the protocol owns contrast the style code is not part of the render, so a
        style reading would describe the encoder alone."""
        out = self._fill(content_leaks=False).probes(include_style=False)
        assert set(out) == {"val_content_contrast_leak", "val_probe_split_by_subject"}

    def test_a_capped_swap_covers_every_ordered_pair_of_contrasts(self) -> None:
        """Truncating the sorted list kept only content rows of the first labels: with three
        contrasts and two rows each, FLAIR (label 2) was never a content source at 12 pairs."""
        acc = DisentanglementAccumulator(per_label_swap_rows=2)
        for label in (0, 0, 1, 1, 2, 2):
            acc.keep(label, (label, f"row{label}"))
        pairs = acc.swap_pairs(max_pairs=12)
        ordered = collections.Counter((a[0], b[0]) for a, b in pairs)
        assert set(ordered) == {(i, j) for i in range(3) for j in range(3) if i != j}
        assert set(ordered.values()) == {2}  # 6 ordered contrast pairs, twice each

    def test_the_content_probe_sees_spatial_structure_not_only_channel_means(self) -> None:
        """A code that keeps the label in its spatial spread, with equal channel means, is a
        leak; spatial means alone read it at chance."""
        labels = _blocked_labels(blocks=8)
        gen = torch.Generator().manual_seed(0)
        content = torch.randn(labels.numel(), 4, 6, 6, generator=gen)
        content = content - content.mean(dim=(2, 3), keepdim=True)  # every channel mean is 0
        # the label's own channel is spread 3x wider: separable one-vs-rest, invisible to means
        gain = torch.ones(labels.numel(), 4, 1, 1)
        gain[torch.arange(labels.numel()), labels] = 3.0
        content = content * gain
        acc = DisentanglementAccumulator()
        acc.add(content, _label_code(labels), labels)
        assert acc.probes()["val_content_contrast_leak"] > 0.4

    def test_swap_pairs_follow_ascending_integer_labels_whatever_the_arrival_order(self) -> None:
        acc = DisentanglementAccumulator(per_label_swap_rows=1)
        for label in (10, 2, 1):  # arrival order 10, 2, 1; a string sort would put 10 before 2
            acc.keep(label, label)
        assert acc.swap_pairs(max_pairs=100)[:2] == [(1, 2), (1, 10)]
