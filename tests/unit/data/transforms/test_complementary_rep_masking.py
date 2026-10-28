"""CoNEX masks: what each mode lets the loss see, and that the input is really masked.

The four modes differ only in ``M2``, so the claims pinned here are the
identifiability conditions the module docstring states, measured over many
scans rather than asserted for one: ``independent`` supervises every supported
line and the overlap of acquired and supervised lines, ``complementary`` never
supervises an acquired non-ACS line, ``same`` never supervises an unacquired one.
"""

from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.transforms.complementary_rep_masking import (
    VALID_MODES,
    ComplementaryRepMasking,
    _acs,
    draw_rep_masks,
)
from spectramr.data.transforms.normalization import acs_band_rows
from spectramr.data.transforms.registry import build_transform, list_transforms

_C, _H, _W, _S = 4, 8, 64, 3
#: Columns outside the acquisition, as M4Raw's partial phase FOV zero-fills them.
_ZERO_COLUMNS = list(range(0, 8)) + list(range(56, 64))


def _subject(file_id: str = "2022061001_T101", depth: int = _S) -> tio.Subject:
    gen = torch.Generator().manual_seed(0)
    data = torch.randn(2 * _C, _H, _W, depth, generator=gen)
    data[:, :, _ZERO_COLUMNS] = 0.0
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=data.clone()),
        target=tio.ScalarImage(tensor=data.clone() + 1.0),
    )
    subject["file_id"] = file_id
    return subject


def _support() -> torch.Tensor:
    support = torch.ones(_W, dtype=torch.bool)
    support[_ZERO_COLUMNS] = False
    return support


def _masks(mode: str, file_id: str, acceleration: float = 2.0) -> tuple[torch.Tensor, ...]:
    return draw_rep_masks(
        _support(),
        acceleration=acceleration,
        center_fraction=0.125,
        mode=mode,
        keys=(file_id, 0, 0),
    )


class TestTheSubjectContract:
    def test_it_is_registered_and_buildable_from_yaml_kwargs(self) -> None:
        assert "complementary_rep_masking" in list_transforms()
        built = build_transform(
            "complementary_rep_masking", acceleration=2.0, center_fraction=0.08, mode="same"
        )
        assert isinstance(built, ComplementaryRepMasking)

    def test_input_is_masked_and_target_is_untouched(self) -> None:
        original = _subject()
        out = ComplementaryRepMasking(2.0, 0.125)(_subject())
        mask = out["mask"].data
        assert mask.shape == (1, _H, _W, _S)
        assert torch.equal(out["input"].data, original["input"].data * mask)
        assert torch.equal(out["target"].data, original["target"].data)

    def test_masks_are_binary_broadcastable_and_inside_the_support(self) -> None:
        out = ComplementaryRepMasking(3.0, 0.125, mode="independent")(_subject())
        for key in ("mask", "loss_mask"):
            values = out[key].data.unique().tolist()
            assert set(values) <= {0.0, 1.0}, key
        m1 = out["mask"].data[0, 0].bool()  # (W, S)
        assert not m1[_ZERO_COLUMNS].any(), "M1 acquired a column outside the support"

    def test_the_support_is_published_beside_the_masks(self) -> None:
        """Its complement is the zero fill, which EI and SSDU score without a reference."""
        out = ComplementaryRepMasking(2.0, 0.125)(_subject())
        support = out["support_mask"].data
        assert support.shape == out["mask"].data.shape
        expected = _support()[None, None, :, None].expand(1, _H, _W, _S).float()
        assert torch.equal(support, expected)
        assert torch.all(out["mask"].data <= support), "M1 escaped the published support"

    def test_structural_zeros_are_supervised_in_every_mode(self) -> None:
        """Otherwise the network may emit unsupervised energy the metrics then read."""
        for mode in sorted(VALID_MODES):
            _, m2 = _masks(mode, "a")
            assert m2[_ZERO_COLUMNS].all(), mode

    def test_it_is_deterministic_per_file_and_differs_across_files(self) -> None:
        a = ComplementaryRepMasking(2.0, 0.125)(_subject("x"))["mask"].data
        b = ComplementaryRepMasking(2.0, 0.125)(_subject("x"))["mask"].data
        c = ComplementaryRepMasking(2.0, 0.125)(_subject("y"))["mask"].data
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_slices_get_their_own_masks(self) -> None:
        mask = ComplementaryRepMasking(2.0, 0.125)(_subject())["mask"].data[0, 0]
        assert not torch.equal(mask[:, 0], mask[:, 1])

    def test_a_slice_record_is_seeded_by_its_volume_index(self) -> None:
        """A depth-1 record for slice 2 must get the mask slice 2 of the volume gets."""
        volume = ComplementaryRepMasking(2.0, 0.125)(_subject())["mask"].data[0, 0]
        record = _subject(depth=1)
        record["slice_index"] = 2
        single = ComplementaryRepMasking(2.0, 0.125)(record)["mask"].data[0, 0, :, 0]
        assert torch.equal(single, volume[:, 2])

    def test_every_mode_feeds_the_same_input(self) -> None:
        """The arms differ in M2 only, so a PSNR gap is attributable to the loss."""
        inputs = [_masks(mode, "same-scan")[0] for mode in sorted(VALID_MODES)]
        assert all(torch.equal(m, inputs[0]) for m in inputs)


class TestTheModes:
    def test_same_supervises_exactly_the_acquired_lines(self) -> None:
        m1, m2 = _masks("same", "a")
        assert torch.equal(m2 & _support(), m1)

    def test_full_target_supervises_everything(self) -> None:
        _, m2 = _masks("full_target", "a")
        assert m2.all()

    def test_complementary_overlaps_only_on_the_acs_at_r2(self) -> None:
        m1, m2 = _masks("complementary", "a", acceleration=2.0)
        support = _support()
        acs = torch.zeros(_W, dtype=torch.bool)
        acs[_W // 2 - 4 : _W // 2 + 4] = True
        assert torch.equal(m1 & m2, acs)
        assert torch.equal((m1 | m2) & support, support)

    def test_complementary_is_rebalanced_above_r2(self) -> None:
        m1, m2 = _masks("complementary", "a", acceleration=4.0)
        assert int((m2 & _support()).sum()) == int(m1.sum())

    def test_acceleration_sets_the_line_count(self) -> None:
        m1, _ = _masks("independent", "a", acceleration=4.0)
        assert int(m1.sum()) == round(int(_support().sum()) / 4.0)

    def test_the_acs_is_in_both_masks(self) -> None:
        for mode in sorted(VALID_MODES):
            m1, m2 = _masks(mode, "a", acceleration=4.0)
            assert m1[_W // 2 - 4 : _W // 2 + 4].all() and m2[_W // 2 - 4 : _W // 2 + 4].all()


class TestIdentifiability:
    """Measured over 200 scans: which lines can the objective ever reach?"""

    @staticmethod
    def _coverage(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        in_loss = torch.zeros(_W, dtype=torch.bool)
        denoised = torch.zeros(_W, dtype=torch.bool)
        for scan in range(200):
            m1, m2 = _masks(mode, f"scan{scan}")
            in_loss |= m2
            denoised |= m1 & m2
        return in_loss & _support(), denoised & _support()

    def test_independent_reaches_every_line_and_denoises_acquired_ones(self) -> None:
        in_loss, denoised = self._coverage("independent")
        assert torch.equal(in_loss, _support())
        assert torch.equal(denoised, _support())

    def test_complementary_never_denoises_an_acquired_non_acs_line(self) -> None:
        in_loss, denoised = self._coverage("complementary")
        assert torch.equal(in_loss, _support())
        acs = torch.zeros(_W, dtype=torch.bool)
        acs[_W // 2 - 4 : _W // 2 + 4] = True
        assert torch.equal(denoised, acs)

    def test_same_never_supervises_an_unacquired_line(self) -> None:
        for scan in range(50):
            m1, m2 = _masks("same", f"scan{scan}")
            assert not (m2 & _support() & ~m1).any()


def test_the_mask_drawer_table_covers_exactly_the_valid_modes() -> None:
    """The second mask is drawn through a table keyed by mode; it must equal VALID_MODES."""
    from spectramr.data.transforms import complementary_rep_masking as mod

    assert set(mod._M2_BY_MODE) == set(mod.VALID_MODES)


class TestItRefuses:
    def test_an_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            ComplementaryRepMasking(2.0, 0.08, mode="complement")

    def test_an_acceleration_below_one(self) -> None:
        with pytest.raises(ValueError, match="acceleration"):
            ComplementaryRepMasking(0.5, 0.08)

    def test_a_subject_without_file_id(self) -> None:
        subject = _subject()
        del subject["file_id"]
        with pytest.raises(ValueError, match="file_id"):
            ComplementaryRepMasking(2.0, 0.08)(subject)


# --- the measurement under its own key (#2324), and the CoNEX arms unmoved -----


def test_the_masked_input_is_also_the_measurement() -> None:
    from spectramr.data.transforms.tio_physics import MEASURED_KSPACE_KEY

    torch.manual_seed(0)
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=torch.randn(8, 16, 32, 1)), file_id="2022_T101"
    )
    out = ComplementaryRepMasking(acceleration=2.0, mode="same")(subject)
    assert torch.equal(out[MEASURED_KSPACE_KEY].data, out["input"].data)


@pytest.mark.parametrize(
    "arm",
    [
        "conex_r2_independent_m4raw",
        "conex_r2_complementary_m4raw",
        "conex_r2_samemask_m4raw",
        "conex_r2_fulltarget_m4raw",
    ],
)
def test_the_conex_objective_does_not_move(arm: str) -> None:
    """Before this key existed the context resolved `measured_kspace` to `kspace` --
    repetition 2 -- and `complex_unet` ignored it. The step must be bit-identical
    with the old resolution and the new one."""
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm, real_batches

    config = load_arm(arm)
    batch, _ = real_batches(config, coils=4, route="m4raw")
    strategy = build_strategy(config)
    torch.manual_seed(0)
    now = strategy._compute_losses_impl(batch.input, batch.target, 0, batch=batch)["loss"]
    batch.metadata.pop("measured_kspace")
    torch.manual_seed(0)
    before = strategy._compute_losses_impl(batch.input, batch.target, 0, batch=batch)["loss"]
    assert torch.equal(now.detach(), before.detach())


def test_the_alternate_repetition_is_masked_with_the_input_mask() -> None:
    """`input_alt` (the second repetition for rep_variance) carries M1 too, so the
    model's output on it estimates the same masked acquisition."""
    subject = _subject()
    subject.add_image(
        tio.ScalarImage(tensor=subject["input"].data.clone() + 1.0, affine=subject["input"].affine),
        "input_alt",
    )
    out = ComplementaryRepMasking(mode="independent")(subject)
    mask = out["mask"].data
    assert torch.equal((out["input_alt"].data != 0).float().amax(dim=0, keepdim=True), mask)
    assert torch.equal(out["input_alt"].data, (subject["input"].data + 1.0) * mask)


@pytest.mark.parametrize("width", [255, 256, 260, 270])
@pytest.mark.parametrize("fraction", [0.0, 0.04, 0.08, 0.125])
def test_the_mask_acs_is_the_band_the_acs_scale_reads(width: int, fraction: float) -> None:
    """``scale_domain: acs`` is mask-invariant only if its band is the mask's ACS (#2612)."""
    rows = acs_band_rows(width, fraction, min_lines=0)
    assert torch.nonzero(_acs(width, fraction)).flatten().tolist() == list(range(rows.start, rows.stop))


# --- cartesian_1d is the historic draw, bit for bit ----------------------------
#
# Hashes taken from the transform before the `pattern` knob existed (dev at
# aa4bd3b3f). The 1-D arms' published numbers were trained on these masks, so a
# change here is a change to every CoNEX/EI/SSDU result, not a refactor.

_GOLDEN_SUPPORT_ZEROS = (31, 226)  # columns [0, 31) and [226, 256) zero-filled: 195 supported
_GOLDEN = {
    (2.0, "independent", 0): ("1cdc295c07cd3b6b", "53b86cdbe9901d85"),
    (2.0, "independent", 1): ("7f7c75f28284718b", "7cbbb434654ea5d0"),
    (2.0, "complementary", 0): ("1cdc295c07cd3b6b", "f2738cf0ff7dbebd"),
    (2.0, "complementary", 1): ("7f7c75f28284718b", "80d1c7f2c5609efc"),
    (2.0, "same", 0): ("1cdc295c07cd3b6b", "80f6716351661366"),
    (2.0, "same", 1): ("7f7c75f28284718b", "5931a8dd8e663ba7"),
    (2.0, "full_target", 0): ("1cdc295c07cd3b6b", "2661920f2409dd6c"),
    (2.0, "full_target", 1): ("7f7c75f28284718b", "2661920f2409dd6c"),
    (4.0, "independent", 0): ("2aad2c6769c349ee", "1c10eca5ffc0fa20"),
    (4.0, "independent", 1): ("5f3d302b1828a763", "06660cae90e13173"),
    (4.0, "complementary", 0): ("2aad2c6769c349ee", "0a9704c32e32fe6b"),
    (4.0, "complementary", 1): ("5f3d302b1828a763", "ae0e35bcf2bf4909"),
    (4.0, "same", 0): ("2aad2c6769c349ee", "d3140f2e0e30bb8c"),
    (4.0, "same", 1): ("5f3d302b1828a763", "2b57ed78f2e87575"),
    (4.0, "full_target", 0): ("2aad2c6769c349ee", "2661920f2409dd6c"),
    (4.0, "full_target", 1): ("5f3d302b1828a763", "2661920f2409dd6c"),
    (8.0, "independent", 0): ("3352462710c941f3", "5ffc41842727e30c"),
    (8.0, "independent", 1): ("434266d241b72204", "369d5ff7105fc591"),
    (8.0, "complementary", 0): ("3352462710c941f3", "da35be4ca463c200"),
    (8.0, "complementary", 1): ("434266d241b72204", "6aee3501d0d5d438"),
    (8.0, "same", 0): ("3352462710c941f3", "2adb43b1fc06d862"),
    (8.0, "same", 1): ("434266d241b72204", "3434d1f88e58a551"),
    (8.0, "full_target", 0): ("3352462710c941f3", "2661920f2409dd6c"),
    (8.0, "full_target", 1): ("434266d241b72204", "2661920f2409dd6c"),
}
_GOLDEN_KEYS = (("2022061001_T101", 0, 0), ("2022061003_T202", 7, 1))


def _digest(mask: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(mask.to(torch.uint8).numpy().tobytes()).hexdigest()[:16]


@pytest.mark.parametrize(("acceleration", "mode", "key"), list(_GOLDEN), ids=str)
def test_cartesian_1d_masks_are_the_historic_masks(acceleration: float, mode: str, key: int) -> None:
    support = torch.ones(256, dtype=torch.bool)
    support[: _GOLDEN_SUPPORT_ZEROS[0]] = False
    support[_GOLDEN_SUPPORT_ZEROS[1] :] = False
    m1, m2 = draw_rep_masks(
        support, acceleration=acceleration, center_fraction=0.08, mode=mode, keys=_GOLDEN_KEYS[key]
    )
    assert (_digest(m1), _digest(m2)) == _GOLDEN[(acceleration, mode, key)]


def test_the_cartesian_1d_subject_is_the_historic_subject() -> None:
    import hashlib

    gen = torch.Generator().manual_seed(0)
    data = torch.randn(8, 16, 64, 3, generator=gen)
    data[:, :, :8] = 0.0
    data[:, :, 56:] = 0.0
    subject = tio.Subject(input=tio.ScalarImage(tensor=data.clone()))
    subject["file_id"] = "2022061001_T101"
    out = ComplementaryRepMasking(3.0, 0.125, mode="independent")(subject)
    assert _digest(out["mask"].data.bool()) == "fd6d52d647ff1cf1"
    assert _digest(out["loss_mask"].data.bool()) == "89116bea3e71767a"
    assert hashlib.sha256(out["input"].data.numpy().tobytes()).hexdigest()[:16] == "723ee0189a5070b0"


# --- the pattern knob ------------------------------------------------------------


def test_an_acs_that_fills_the_budget_raises_instead_of_lowering_r() -> None:
    """R16 on M4Raw's ~195 columns is 12 lines against a 20-line ACS: the old draw
    returned the ACS alone (R~9.75) without a word."""
    support = torch.ones(256, dtype=torch.bool)
    support[:31] = False
    support[226:] = False
    with pytest.raises(ValueError, match=r"ACS alone at R=9\.75"):
        draw_rep_masks(
            support, acceleration=16.0, center_fraction=0.08, mode="same", keys=("a", 0, 0)
        )


def test_an_unknown_pattern_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="Unknown sampling pattern"):
        ComplementaryRepMasking(4.0, 0.08, pattern="hexagonal")


@pytest.mark.parametrize("pattern", ["poisson_disk", "radial"])
def test_a_two_dimensional_pattern_masks_the_subject(pattern: str) -> None:
    from spectramr.data.transforms.tio_physics import REALISED_ACCELERATION_KEY

    gen = torch.Generator().manual_seed(0)
    data = torch.randn(8, 64, 64, 2, generator=gen)
    data[:, :, :8] = 0.0
    data[:, :, 56:] = 0.0
    subject = tio.Subject(input=tio.ScalarImage(tensor=data.clone()))
    subject["file_id"] = "2022061001_T101"
    out = ComplementaryRepMasking(4.0, 0.125, mode="independent", pattern=pattern)(subject)
    mask = out["mask"].data
    assert mask.shape == (1, 64, 64, 2)
    assert not torch.equal(mask[0, 0], mask[0, 16])  # varies along the readout too
    assert torch.equal(out["input"].data, data * mask)
    assert float(out[REALISED_ACCELERATION_KEY]) == pytest.approx(4.0, rel=0.05)


def test_the_cartesian_draw_stamps_its_realised_r() -> None:
    from spectramr.data.transforms.tio_physics import REALISED_ACCELERATION_KEY

    out = ComplementaryRepMasking(3.0, 0.125)(_subject())
    assert float(out[REALISED_ACCELERATION_KEY]) == pytest.approx(48 / 16)
