"""Non-finite guard in ``ImageCollateStrategy.collate`` (finding M3).

The guard used to log at ERROR and then ``torch.nan_to_num(value, nan=0.0)``,
returning the batch anyway — training continued on fabricated zeros, which the
dataset layer explicitly declines to do (``data/datasets/contrast_aware.py``
refuses the identical substitution four times over).

Infinities were mishandled two different ways, both silent, because the old guard
was gated on ``torch.isnan(value).any()``:

* tensor carries inf but no NaN -> the guard never fired, so the inf was neither
  logged nor touched and flowed into the model unchecked;
* tensor carries both -> ``nan_to_num`` fired, and since ``posinf``/``neginf``
  default to ``None`` (meaning "replace", not "leave alone") the inf was rewritten
  to the dtype's largest finite value, ~3.4e38 for float32, while the log line
  spoke only of NaNs.

It now raises ``ValueError`` on either. ``data.collation.validate_nans: false``
remains the opt-out and means "pass values through unmodified", never
"silently repair".
"""

import pytest
import torch

from spectramr.data.collation.strategies import ImageCollateStrategy


class TestNonFiniteGuardRaises:
    """A non-finite value must abort the batch, not be repaired behind the caller."""

    def test_nan_in_float_tensor_raises_and_names_the_key(self):
        """The offending key must be named, so the corrupt volume is findable."""
        strategy = ImageCollateStrategy()
        batch = [
            {"target": torch.tensor([1.0, 2.0])},
            {"target": torch.tensor([float("nan"), 4.0])},
        ]

        with pytest.raises(ValueError) as excinfo:
            strategy.collate(batch)

        msg = str(excinfo.value)
        assert "'target'" in msg, msg
        # Batch size and dtype are part of the reported context.
        assert "batch of 2 sample(s)" in msg, msg
        assert "torch.float32" in msg, msg

    def test_message_reports_nan_and_inf_counts_separately(self):
        """NaN and inf are distinct corruptions; a single total would conflate them.

        This mixed tensor is the case where the old code did its quietest damage:
        the NaN gate fired, so ``nan_to_num(value, nan=0.0)`` ran and clamped the
        inf to ~3.4e38 as a side effect, while the ERROR line reported NaNs only.
        Verified against the pre-fix expression: the inf becomes 3.4028235e+38.
        """
        strategy = ImageCollateStrategy()
        batch = [
            {"kspace": torch.tensor([float("nan"), float("nan"), float("inf"), 1.0])}
        ]

        with pytest.raises(ValueError) as excinfo:
            strategy.collate(batch)

        msg = str(excinfo.value)
        assert "3/4 non-finite" in msg, msg
        assert "2 NaN" in msg, msg
        assert "1 +/-inf" in msg, msg

    @pytest.mark.parametrize("sign", [1.0, -1.0], ids=["posinf", "neginf"])
    def test_infinity_without_any_nan_raises(self, sign):
        """Regression: a pure-inf tensor used to escape the guard entirely.

        The old check was gated on ``torch.isnan(value).any()``, which is False
        here, so nothing was logged and nothing was replaced — +/-inf flowed
        straight into the model. Confirmed against the pre-fix expression:
        ``isnan().any()`` is False for this batch, so the guard never fired.
        """
        strategy = ImageCollateStrategy()
        batch = [{"input": torch.tensor([1.0, sign * float("inf")])}]

        with pytest.raises(ValueError) as excinfo:
            strategy.collate(batch)

        msg = str(excinfo.value)
        assert "'input'" in msg, msg
        assert "0 NaN" in msg, msg
        assert "1 +/-inf" in msg, msg

    def test_nan_in_complex_tensor_raises(self):
        """Complex k-space is the common case here, and ``isfinite`` covers it."""
        strategy = ImageCollateStrategy()
        batch = [
            {
                "kspace": torch.tensor(
                    [complex(float("nan"), 0.0), complex(1.0, 2.0)],
                    dtype=torch.complex64,
                )
            }
        ]

        with pytest.raises(ValueError) as excinfo:
            strategy.collate(batch)

        msg = str(excinfo.value)
        assert "'kspace'" in msg, msg
        assert "1 NaN" in msg, msg


class TestNonFiniteGuardDoesNotOverreach:
    """The guard must not fire on clean data, nor on dtypes it cannot apply to."""

    def test_validate_nans_false_passes_nan_through_unmodified(self):
        """Opting out means pass-through, NOT silent repair.

        Asserting NaN on the OUTPUT is the whole point: a zero here would mean the
        opt-out still fabricated data, just without saying so.
        """
        strategy = ImageCollateStrategy(validate_nans=False)
        batch = [{"target": torch.tensor([float("nan"), 2.0])}]

        result = strategy.collate(batch)

        assert torch.isnan(
            result["target"]
        ).any(), "validate_nans=False must pass the NaN through, not repair it"
        assert result["target"].shape == (1, 2)
        # The finite neighbour is untouched too.
        assert result["target"][0, 1].item() == 2.0

    def test_validate_nans_false_passes_inf_through_unmodified(self):
        """The opt-out must not quietly clamp inf to ~3.4e38 either."""
        strategy = ImageCollateStrategy(validate_nans=False)
        batch = [{"target": torch.tensor([float("inf"), 2.0])}]

        result = strategy.collate(batch)

        assert torch.isinf(
            result["target"]
        ).any(), "validate_nans=False must pass the inf through, not clamp it"

    def test_finite_values_whose_sum_overflows_are_not_refused(self):
        """The one-reduction fast path overflows here; the exact pass must clear it."""
        strategy = ImageCollateStrategy()
        big = torch.finfo(torch.float32).max
        result = strategy.collate([{"target": torch.tensor([big, big])}])
        assert torch.isfinite(result["target"]).all()

    def test_integer_tensor_is_not_checked_and_does_not_raise(self):
        """The guard skips non-float, non-complex dtypes.

        An int64 cannot hold NaN/inf, so ``iinfo.max`` is the closest it gets to
        "looks non-finite" — it must survive with its exact value and dtype, and
        must not raise, alongside a clean float in the same batch.
        """
        strategy = ImageCollateStrategy()
        long_max = torch.iinfo(torch.long).max
        batch = [
            {
                "image": torch.tensor([1.0, 2.0]),
                "contrast_idx": torch.tensor(long_max, dtype=torch.long),
            },
            {
                "image": torch.tensor([3.0, 4.0]),
                "contrast_idx": torch.tensor(0, dtype=torch.long),
            },
        ]

        result = strategy.collate(batch)

        assert result["contrast_idx"].dtype == torch.long
        assert result["contrast_idx"][0].item() == long_max
        assert result["contrast_idx"][1].item() == 0

    def test_clean_batch_collates_unchanged(self):
        """Guard against over-tightening: finite data must pass through untouched."""
        strategy = ImageCollateStrategy()
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        second = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        batch = [{"image": first}, {"image": second}]

        result = strategy.collate(batch)

        assert result["image"].shape == (2, 2, 2)
        assert torch.equal(result["image"], torch.stack([first, second]))


class TestChannelCountsAreNotPaddable:
    """B4. Channels are PHYSICAL — coil c of a 20-coil scan is a different
    element than coil c of a 16-coil scan.

    Centre-padding was worse than merely wrong: a 16-coil sample padded to 20
    got 2 zero channels BEFORE and 2 after, so its coil 0 landed at index 2 and
    was stacked against coil 2 of the 20-coil sample. Verified before the fix:

        stacked channel sums = [0, 0, 64, 64, ..., 64, 0, 0]

    ``m4raw_dataset._select_consistent_reps`` already refuses exactly this for
    NEX repetitions; collation did it anyway.
    """

    @staticmethod
    def _collate(batch):
        from spectramr.data.collation.strategies import CollateStrategyFactory

        return CollateStrategyFactory.create("image").collate(batch)

    @staticmethod
    def _sample(channels, size=8):
        import torch

        return {
            "input": torch.ones(channels, size, size),
            "target": torch.ones(channels, size, size),
        }

    def test_mixed_channel_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="mixes channel counts"):
            self._collate([self._sample(16), self._sample(20)])

    def test_the_message_explains_why_padding_is_not_the_answer(self) -> None:
        """A shape error that only says "shapes differ" invites someone to pad
        harder. This one has to say the channels are not interchangeable."""
        with pytest.raises(ValueError) as exc:
            self._collate([self._sample(16), self._sample(20)])
        message = str(exc.value)
        assert "physical" in message
        assert "SHIFTS" in message  # the centre-padding index shift
        assert "compress coils" in message  # an actionable way out

    def test_uniform_channels_still_collate(self) -> None:
        out = self._collate([self._sample(4), self._sample(4)])
        assert tuple(out["input"].shape) == (2, 4, 8, 8)


class TestSpatialPaddingIsRecorded:
    """B4's other half. ``unpad()`` has always taken a ``padding_info``
    argument and NOTHING ever produced one — so the padding was irreversible
    and invisible. Val metrics on a padded cohort are computed partly over
    fabricated zero regions (PSNR inflated) with nothing in the batch saying so.
    """

    @staticmethod
    def _collate(batch):
        from spectramr.data.collation.strategies import CollateStrategyFactory

        return CollateStrategyFactory.create("image").collate(batch)

    @staticmethod
    def _sample(size):
        import torch

        return {
            "input": torch.ones(2, size, size),
            "target": torch.ones(2, size, size),
        }

    def test_padding_is_emitted_when_it_happens(self) -> None:
        out = self._collate([self._sample(8), self._sample(12)])
        assert "input_padding" in out
        assert "target_padding" in out

    def test_the_record_is_what_unpad_consumes(self) -> None:
        """Per-sample, ``None`` for the untouched ones — the exact shape
        ``unpad(padded, padding_info)`` expects, so the pair is finally usable
        end to end."""
        out = self._collate([self._sample(8), self._sample(12)])
        record = out["input_padding"]
        assert len(record) == 2
        assert record[1] is None  # the largest sample was not padded
        assert record[0] is not None
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in record[0])

    def test_a_uniform_batch_records_nothing(self) -> None:
        """No key when no padding — an always-present empty record would read
        as "padding happened" to anything scanning the batch."""
        out = self._collate([self._sample(8), self._sample(8)])
        assert "input_padding" not in out

    def test_the_padded_result_is_unchanged_in_shape(self) -> None:
        """Recording is additive: the stacked tensor is what it always was."""
        out = self._collate([self._sample(8), self._sample(12)])
        assert tuple(out["input"].shape) == (2, 2, 12, 12)


class TestTheKeySetIsCheckedAcrossTheBatch:
    """B13. ``for key in batch[0]`` silently dropped any key the FIRST sample
    happened not to carry — and the samples that do carry it are the
    interesting ones (a dataset attaching `sensitivity` only when coil maps
    exist, a transform adding `scout` conditionally). Nothing errored; the
    mechanism behind the key simply never received its input.
    """

    @staticmethod
    def _collate(batch):
        from spectramr.data.collation.strategies import CollateStrategyFactory

        return CollateStrategyFactory.create("image").collate(batch)

    @staticmethod
    def _sample(**extra):
        import torch

        base = {"input": torch.ones(2, 8, 8), "target": torch.ones(2, 8, 8)}
        base.update(extra)
        return base

    def test_a_key_missing_from_one_sample_raises(self) -> None:
        import torch

        with pytest.raises(ValueError, match="disagree about which keys"):
            self._collate(
                [self._sample(sensitivity=torch.ones(2, 8, 8)), self._sample()]
            )

    def test_it_fires_even_when_the_first_sample_is_the_poor_one(self) -> None:
        """The direction the old code was blind to: `batch[0]` lacking the key
        meant it was dropped for everybody, with no error at all."""
        import torch

        with pytest.raises(ValueError, match="disagree about which keys"):
            self._collate(
                [self._sample(), self._sample(sensitivity=torch.ones(2, 8, 8))]
            )

    def test_the_message_counts_the_affected_samples(self) -> None:
        import torch

        with pytest.raises(ValueError) as exc:
            self._collate(
                [self._sample(scout=torch.ones(2, 8, 8)), self._sample(), self._sample()]
            )
        assert "2/3 samples missing" in str(exc.value)
        assert "robust" in str(exc.value)  # the ragged-batch escape

    def test_a_consistent_batch_keeps_every_key(self) -> None:
        import torch

        out = self._collate(
            [
                self._sample(sensitivity=torch.ones(2, 8, 8)),
                self._sample(sensitivity=torch.ones(2, 8, 8)),
            ]
        )
        assert "sensitivity" in out


class TestSlabTargetModeIsReachable:
    """B6. `CollationStrategySelector` hardcoded `target_mode = "middle"`, so an
    arm could not choose among the three modes `SlabCollateStrategy` already
    implemented. The default is unchanged, so no existing run moves — what
    changes is that the other two become selectable at all.
    """

    @staticmethod
    def _select(**kw):
        from spectramr.config.schemas.collation import CollationConfigSchema
        from spectramr.data.collation.strategy_selector import (
            CollationStrategySelector,
        )

        return CollationStrategySelector.select_strategy(
            config=CollationConfigSchema(strategy="slab", **kw),
            dataset_type="nifti",
            enable_slab_mode=True,
            patch_size=(64, 64, 8),
        )

    def test_the_default_is_unchanged(self) -> None:
        assert self._select()[1]["target_mode"] == "middle"

    def test_the_config_value_reaches_the_strategy(self) -> None:
        assert self._select(slab_target_mode="flatten")[1]["target_mode"] == "flatten"
        assert self._select(slab_target_mode="keep")[1]["target_mode"] == "keep"

    def test_the_schema_rejects_an_unknown_mode(self) -> None:
        import pydantic

        from spectramr.config.schemas.collation import CollationConfigSchema

        with pytest.raises(pydantic.ValidationError):
            CollationConfigSchema(slab_target_mode="middl")


class TestSlabTargetModeDispatch:
    """`keep` used to be reached by FALLING OFF the if/elif, so an unknown
    value silently kept as well — the arm asked for the centre slice and was
    handed the whole slab, with nothing said (#9)."""

    @staticmethod
    def _collate(mode):
        import torch

        from spectramr.data.collation.strategies import SlabCollateStrategy

        batch = [
            {
                "input": torch.ones(2, 8, 8, 6),
                "target": torch.ones(2, 8, 8, 6),
            }
            for _ in range(2)
        ]
        return SlabCollateStrategy(target_mode=mode).collate(batch)

    def test_middle_keeps_one_depth_slice(self) -> None:
        out = self._collate("middle")
        assert out["target"].ndim == 4

    def test_flatten_supervises_every_slice(self) -> None:
        out = self._collate("flatten")
        assert out["target"].shape[1] == 2 * 6  # C * D

    def test_keep_leaves_the_volume_intact(self) -> None:
        out = self._collate("keep")
        assert out["target"].ndim == 5

    def test_an_unknown_mode_raises_instead_of_keeping(self) -> None:
        with pytest.raises(ValueError, match="Unknown slab target_mode"):
            self._collate("middl")


class TestFlatten3dTo2dStaysUnreachable:
    """A11. Neither delete nor wire — record why it must stay off.

    Deleting removes `squeezing_collate`, a documented helper whose shape
    contract has its own integration test class. WIRING it would double-flatten:
    `pipelines/train.py` already performs the production 5D->4D validation
    flatten, so an arm that set the knob would flatten twice — the same class of
    defect as the double-normalization (#760) and the double-FFT (A4).
    """

    def test_it_is_not_a_schema_field(self) -> None:
        from spectramr.config.schemas.collation import CollationConfigSchema

        assert "flatten_3d_to_2d" not in CollationConfigSchema.model_fields

    def test_the_selector_never_passes_it(self) -> None:
        import inspect

        from spectramr.data.collation.strategy_selector import (
            CollationStrategySelector,
        )

        src = inspect.getsource(CollationStrategySelector._build_strategy_kwargs)
        assert "flatten_3d_to_2d" not in src

    def test_the_production_flatten_lives_in_the_train_pipeline(self) -> None:
        """The capability is not missing — it is somewhere else, which is why
        exposing this one would duplicate it."""
        from pathlib import Path

        train = (
            Path(__file__).resolve().parents[4]
            / "src/spectramr/pipelines/train.py"
        )
        assert "5D→4D" in train.read_text(encoding="utf-8")

    def test_the_standalone_helper_still_works(self) -> None:
        """Unreachable from config is not the same as dead."""
        from spectramr.data.collation.strategies import squeezing_collate

        assert callable(squeezing_collate)
