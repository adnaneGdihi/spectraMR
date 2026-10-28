"""``equivariant_imaging_ops``: which declared k-space loss the EI anchor calls.

The one helper here without a model: ``anchor_scores_kspace_losses`` is what the
loss-domain audit asks before it calls an image arm's ``kspace_losses`` unbridged.
"""

from __future__ import annotations

import types

import pytest

from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
    anchor_scores_kspace_losses,
)


class TestTheZeroFilledBaselineOnTheCoilMapFooting:
    """The RSS val_zf_* columns could not be read against sense_psnr or the corrected MSE."""

    @staticmethod
    def _scene(accel: int):
        import torch

        from spectramr.core.metrics.coil_manifold import as_interleaved
        from spectramr.data.batch_types import BatchAdapter
        from spectramr.infrastructure.physics.fft_ops import sense_forward
        from spectramr.infrastructure.physics.multicoil_ei_operator import MulticoilEIOperator

        gen = torch.Generator().manual_seed(0)
        maps = torch.complex(torch.randn(1, 4, 32, 32, generator=gen), torch.randn(1, 4, 32, 32, generator=gen))
        maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
        image = torch.zeros(1, 1, 32, 32, dtype=torch.complex64)
        image[..., 8:24, 8:24] = 0.8
        full = sense_forward(image, maps, None)
        mask = torch.zeros(1, 1, 32, 32)
        mask[..., ::accel] = 1.0
        mask[..., 14:18] = 1.0
        op = MulticoilEIOperator(maps, mask, where="in the test")
        context = {"measured_kspace": as_interleaved(full * mask)}
        batch = BatchAdapter.from_dict(
            {"input": context["measured_kspace"], "target": as_interleaved(full),
             "sensitivity_complex": maps[..., None]}
        )
        return op, context, as_interleaved(full), batch

    @staticmethod
    def _computer(names: list[str]):
        from spectramr.core.metrics.computer import ValidationMetricsComputer
        from spectramr.core.metrics.types import ValidationMetricsConfig

        computer = ValidationMetricsComputer(ValidationMetricsConfig.from_dict({"metrics": names}))
        computer.data_range = 1.0
        return computer

    def test_the_columns_are_graded_and_fall_with_acceleration(self) -> None:
        from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
            zero_filled_context_metrics,
        )

        computer = self._computer(["sense_psnr", "sense_ssim", "reference_noise_corrected_mse"])
        scores = {}
        for accel in (1, 4):
            op, context, target, batch = self._scene(accel)
            scores[accel] = zero_filled_context_metrics(computer, context, target, op, batch)
        assert set(scores[4]) == {
            "val_zf_sense_psnr", "val_zf_sense_ssim", "val_zf_reference_noise_corrected_mse"
        }
        # Fully sampled and noise-free, A^H y is the truth on the support.
        assert scores[1]["val_zf_reference_noise_corrected_mse"] < 1e-6
        assert scores[4]["val_zf_sense_psnr"] < scores[1]["val_zf_sense_psnr"]
        assert scores[4]["val_zf_reference_noise_corrected_mse"] > 1e-4

    def test_only_the_configured_columns_are_emitted(self) -> None:
        from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
            zero_filled_context_metrics,
        )

        op, context, target, batch = self._scene(4)
        out = zero_filled_context_metrics(self._computer(["psnr", "sense_psnr"]), context, target, op, batch)
        assert set(out) == {"val_zf_sense_psnr"}
        assert zero_filled_context_metrics(self._computer(["sense_psnr"]), context, target, None, batch) == {}

    def test_under_sense_magnitude_the_baseline_is_graded_on_the_sense_pair(self) -> None:
        from spectramr.core.metrics.sense_image_metrics import (
            SenseMapsMissingError,
            sense_scoring_pair,
        )
        from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
            zero_filled_metrics,
        )
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            scoring_maps,
        )

        computer = self._computer(["psnr", "clinical_ssim"])
        names = ("psnr", "clinical_ssim")
        op, context, target, batch = self._scene(4)
        out = zero_filled_metrics(computer, context, target, op, names, transform="sense_magnitude", batch_data=batch)
        assert set(out) == {"val_zf_psnr", "val_zf_clinical_ssim"}
        image = op.model_input(context["measured_kspace"])
        zf, ref = (t.abs() for t in sense_scoring_pair(image, target, scoring_maps(batch, image)[0]))
        assert out["val_zf_psnr"] == pytest.approx(computer.compute(zf, ref, only=("psnr",))["psnr"])
        # Fully sampled and noise-free, A^H y is the SENSE target itself.
        op1, context1, target1, batch1 = self._scene(1)
        full = zero_filled_metrics(computer, context1, target1, op1, names, transform="sense_magnitude", batch_data=batch1)
        assert full["val_zf_psnr"] > out["val_zf_psnr"] + 10
        with pytest.raises(ValueError, match="no multi-coil operator"):
            zero_filled_metrics(computer, context, target, None, names, transform="sense_magnitude", batch_data=batch)
        with pytest.raises(SenseMapsMissingError):
            zero_filled_metrics(computer, context, target, op, names, transform="sense_magnitude")


def _config(target: str | None) -> types.SimpleNamespace:
    ei = None if target is None else types.SimpleNamespace(measurement_target=target)
    return types.SimpleNamespace(training=types.SimpleNamespace(equivariant_imaging=ei))


@pytest.mark.parametrize(
    ("target", "names", "scored"),
    [
        ("alternate_repetition", ["masked_kspace_l2"], True),
        ("measured", ["masked_kspace_l2"], False),
        (None, ["masked_kspace_l2"], False),
        ("alternate_repetition", ["kspace_l1_sparsity"], False),
        ("alternate_repetition", ["masked_kspace_l2", "masked_kspace_l2"], False),
        ("alternate_repetition", ["not_a_registered_loss"], False),
        ("alternate_repetition", [], False),
    ],
    ids=["anchor", "measured", "not-ei", "other-loss", "two-entries", "unknown", "empty"],
)
def test_only_the_alternate_repetition_anchor_scores_its_one_loss(
    target: str | None, names: list[str], scored: bool
) -> None:
    assert anchor_scores_kspace_losses(_config(target), names) is scored


def test_an_image_network_is_read_on_the_maps_support_only() -> None:
    """The EI branches and the validation image see x_hat zeroed where S = 0."""
    import torch

    from spectramr.infrastructure.physics.multicoil_ei_operator import MulticoilEIOperator
    from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
        network_image,
    )

    maps = torch.ones(1, 2, 8, 8, dtype=torch.complex64) / 2**0.5
    maps[..., :2] = 0.0
    op = MulticoilEIOperator(maps, torch.ones(1, 1, 8, 8), where="in the test")
    output = torch.full((1, 2, 8, 8), 3.0)
    image = network_image(output, op, emits_kspace=False)
    assert torch.all(image[..., :2] == 0) and torch.all(image[..., 2:] == 3.0)
    assert network_image(output, None, emits_kspace=False) is output
