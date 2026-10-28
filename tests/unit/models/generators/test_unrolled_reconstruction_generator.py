"""Regression tests for ``RegularizationBlock`` activation dispatch.

Pitfall #9/#10: ``RegularizationBlock`` previously appended a nonlinearity only
for ``activation in {'relu', 'leaky_relu'}`` with no else branch, so any other
value (typo 'gelu'/'silu') silently produced a purely-linear Conv2d+BatchNorm2d
stack with ZERO nonlinearity — a catastrophic silent degradation that still
passes smoke. Post-fix: an unknown activation RAISES, and the two valid values
still insert exactly one nonlinearity per layer (byte-identical numerics).
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from spectramr.models.generators.unrolled_reconstruction_generator import (
    RegularizationBlock,
    UnrolledReconstructionGenerator,
)


def test_regularization_block_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        RegularizationBlock(activation="gelu")


def test_unrolled_generator_rejects_unknown_activation() -> None:
    # The bad activation propagates through to RegularizationBlock construction.
    with pytest.raises(ValueError, match="Unknown activation"):
        UnrolledReconstructionGenerator(activation="silu")


@pytest.mark.parametrize(
    "activation,expected",
    [("relu", nn.ReLU), ("leaky_relu", nn.LeakyReLU)],
)
def test_valid_activation_inserts_nonlinearity(activation, expected) -> None:
    block = RegularizationBlock(
        in_channels=1, out_channels=1, features=8, num_layers=3, activation=activation
    )
    modules = list(block.model)
    # Regression guard: a valid activation must yield a non-empty nonlinearity set
    # (the bug produced a purely-linear Conv+BN stack).
    acts = [m for m in modules if isinstance(m, (nn.ReLU, nn.LeakyReLU))]
    assert acts, "valid activation must insert a nonlinearity"
    assert all(isinstance(m, expected) for m in acts)


class TestVariationalNetworkOutputActivation:
    """Pitfall #9 for ``VariationalNetworkGenerator.output_activation``: the
    documented None/'none' -> Identity path stays, but an unrecognized STRING
    must raise instead of silently becoming Identity (a typo'd 'tanh' would
    otherwise ship an unbounded output while the YAML advertises a bounded
    one)."""

    @staticmethod
    def _build(output_activation):
        from spectramr.models.generators.unrolled_reconstruction_generator import (
            VariationalNetworkGenerator,
        )

        return VariationalNetworkGenerator(
            in_channels=2,
            out_channels=2,
            img_size=(16, 16),
            num_stages=1,
            features=8,
            output_activation=output_activation,
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, nn.Identity),
            ("none", nn.Identity),
            ("NONE", nn.Identity),
            ("tanh", nn.Tanh),
            ("Tanh", nn.Tanh),
            ("sigmoid", nn.Sigmoid),
            ("relu", nn.ReLU),
        ],
    )
    def test_valid_output_activation_resolves(self, value, expected):
        gen = self._build(value)
        assert isinstance(gen.output_activation, expected)

    @pytest.mark.parametrize("value", ["swish", "identity", "leaky_relu"])
    def test_unknown_output_activation_raises(self, value):
        with pytest.raises(ValueError, match="'none', 'tanh', 'sigmoid', 'relu'"):
            self._build(value)


class TestTheUnrolledModelCanRunDataConsistency:
    """#2327 blocker 2: DC returned complex, assumed unbatched maps, ignored num_coils."""

    @staticmethod
    def _inputs(batch: int = 2, coils: int = 4, size: int = 16):
        import torch

        torch.manual_seed(0)
        maps = torch.randn(batch, coils, size, size, dtype=torch.complex64)
        maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
        mask = torch.zeros(batch, 1, size, size)
        mask[..., ::2, :] = 1.0
        kspace = torch.randn(batch, coils, size, size, dtype=torch.complex64) * mask
        return torch.randn(batch, 2, size, size), kspace, mask, maps

    def test_a_multicoil_forward_runs_and_stays_real_interleaved(self) -> None:
        model = UnrolledReconstructionGenerator(
            in_channels=2,
            out_channels=2,
            img_size=(16, 16),
            num_unrolls=2,
            features=8,
            num_coils=4,
            output_activation="none",
        )
        x, kspace, mask, maps = self._inputs()
        out = model(x, kspace_measured=kspace, mask=mask, sensitivity_maps=maps)
        assert out.shape == (2, 2, 16, 16) and not out.is_complex()

    def test_a_coil_count_the_model_was_not_built_for_raises(self) -> None:
        model = UnrolledReconstructionGenerator(
            in_channels=2, out_channels=2, img_size=(16, 16), num_unrolls=1, features=8, num_coils=1
        )
        x, kspace, mask, maps = self._inputs()
        with pytest.raises(ValueError, match="num_coils=1"):
            model(x, kspace_measured=kspace, mask=mask, sensitivity_maps=maps)

    def test_multicoil_dc_without_maps_raises(self) -> None:
        model = UnrolledReconstructionGenerator(
            in_channels=2, out_channels=2, img_size=(16, 16), num_unrolls=1, features=8, num_coils=4
        )
        x, kspace, mask, _ = self._inputs()
        with pytest.raises(ValueError, match="needs sensitivity_maps"):
            model(x, kspace_measured=kspace, mask=mask)

    @pytest.mark.parametrize("missing", ["kspace_measured", "mask"])
    def test_a_half_supplied_dc_pair_raises(self, missing) -> None:
        model = UnrolledReconstructionGenerator(
            in_channels=2, out_channels=2, img_size=(16, 16), num_unrolls=1, features=8, num_coils=4
        )
        x, kspace, mask, maps = self._inputs()
        given = {"kspace_measured": kspace, "mask": mask}
        given.pop(missing)
        with pytest.raises(ValueError, match="needs both kspace_measured and mask"):
            model(x, sensitivity_maps=maps, **given)

    def test_image_only_forward_still_runs(self) -> None:
        model = UnrolledReconstructionGenerator(
            in_channels=2, out_channels=2, img_size=(16, 16), num_unrolls=1, features=8, num_coils=1
        )
        out = model(self._inputs()[0])
        assert out.shape == (2, 2, 16, 16)

    @pytest.mark.parametrize(("name", "expected"), [("tanh", nn.Tanh), ("none", nn.Identity)])
    def test_the_output_activation_is_declared(self, name, expected) -> None:
        model = UnrolledReconstructionGenerator(num_unrolls=1, features=8, output_activation=name)
        assert isinstance(model.output_activation, expected)

    def test_the_default_keeps_what_every_existing_arm_ran(self) -> None:
        assert isinstance(UnrolledReconstructionGenerator(num_unrolls=1).output_activation, nn.Tanh)

    def test_an_unknown_output_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown output_activation"):
            UnrolledReconstructionGenerator(num_unrolls=1, output_activation="softplus")


def test_varnet_does_not_turn_a_nan_input_into_zeros() -> None:
    import torch

    from spectramr.models.generators.unrolled_reconstruction_generator import (
        VariationalNetworkGenerator,
    )

    model = VariationalNetworkGenerator(
        in_channels=2, out_channels=2, img_size=(16, 16), num_stages=1, features=8
    ).eval()
    x = torch.randn(1, 2, 16, 16)
    x[0, 0, 0, 0] = float("nan")
    with torch.no_grad():
        out = model(x)
    assert torch.isnan(out).any()
