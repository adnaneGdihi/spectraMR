"""The operator an Equivariant Imaging arm inverts, and why the choice decides the arm.

EI recovers what a *single* operator ``A`` leaves unidentifiable: it assumes the
signal set is invariant under a group and uses the orbit ``{A T_g}`` to pin down
what ``A`` alone cannot. That premise is a property of the operator, so an arm
that compresses its coils away is not tuning a hyperparameter -- it is
manufacturing the under-determination its assumed symmetry then repairs.

The identifiability test below is the load-bearing one: it is the measurement
that motivates the whole module, and it needs no data or GPU.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.multicoil_ei_operator import MulticoilEIOperator

_B, _C, _S = 2, 4, 32


def _maps(batch: int = _B, coils: int = _C, size: int = _S) -> torch.Tensor:
    torch.manual_seed(0)
    smaps = (torch.randn(batch, coils, size, size) + 1j * torch.randn(batch, coils, size, size)).to(
        torch.complex64
    )
    return smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt().clamp(min=1e-6)


def _mask(size: int = _S, accel: int = 4) -> torch.Tensor:
    mask = torch.zeros(1, 1, size, size)
    mask[..., ::accel, :] = 1.0
    return mask


def _kspace() -> torch.Tensor:
    torch.manual_seed(1)
    return (torch.randn(_B, _C, _S, _S) + 1j * torch.randn(_B, _C, _S, _S)).to(torch.complex64)


class TestTheOperatorDecidesWhetherEIHasAProblem:
    """Why this module exists, as a number rather than an argument."""

    @staticmethod
    def _unidentifiable(coils: int, accel: int, pixels: int = 16) -> tuple[int, int]:
        """``(unidentifiable dims, signal dims)`` for ``A = M F`` per coil.

        The physical signal set is ``{x : x_n in span(s_n)}`` -- one complex
        scalar per pixel -- so its dimension is the pixel count whatever the coil
        count is. What changes is how much of it ``A`` preserves.
        """
        torch.manual_seed(0)
        fourier = torch.fft.fft(torch.eye(pixels, dtype=torch.complex128), norm="ortho")
        keep = torch.zeros(pixels)
        keep[::accel] = 1.0
        per_coil = torch.diag(keep.to(torch.complex128)) @ fourier
        forward = torch.block_diag(*[per_coil] * coils)

        sens = torch.randn(coils, pixels, dtype=torch.complex128)
        sens = sens / sens.norm(dim=0, keepdim=True)
        basis = torch.zeros(coils * pixels, pixels, dtype=torch.complex128)
        for n in range(pixels):
            for c in range(coils):
                basis[c * pixels + n, n] = sens[c, n]

        singular = torch.linalg.svdvals(forward @ basis)
        return pixels - int((singular > 1e-9).sum()), pixels

    def test_one_virtual_coil_at_r4_is_badly_underdetermined(self) -> None:
        """What `ei_unet_m4raw_r4` inverts: 12 of 16 directions unrecoverable."""
        unidentifiable, total = self._unidentifiable(coils=1, accel=4)
        assert (unidentifiable, total) == (12, 16)

    def test_four_physical_coils_at_r4_are_fully_determined(self) -> None:
        """What this module lets an arm invert instead: nothing left over.

        This is the finding: at R4 the coils M4Raw already acquired determine
        the image exactly, so a symmetry assumed of brain anatomy is repairing
        damage the arm did to itself.
        """
        unidentifiable, _ = self._unidentifiable(coils=4, accel=4)
        assert unidentifiable == 0

    def test_the_gain_is_not_an_artefact_of_one_acceleration(self) -> None:
        """Four coils help until R exceeds the coil count, then they cannot."""
        assert self._unidentifiable(coils=4, accel=2)[0] == 0
        assert self._unidentifiable(coils=4, accel=8)[0] == 8


class TestItIsAnAdjointPair:
    """A wrong ``A^H`` trains happily and reconstructs the wrong operator."""

    def test_the_adjoint_identity_holds(self) -> None:
        """``<Ax, y> == <x, A^H y>`` -- the test a plausible-but-wrong A^H fails."""
        operator = MulticoilEIOperator(_maps(), _mask(), where="in the test")
        torch.manual_seed(2)
        image = (torch.randn(_B, 1, _S, _S) + 1j * torch.randn(_B, 1, _S, _S)).to(torch.complex64)
        kspace = _kspace()
        interleaved = operator.model_input(kspace)
        adjoint = torch.complex(interleaved[:, :1], interleaved[:, 1:])
        lhs = (operator.forward(image).conj() * kspace).sum()
        rhs = (image.conj() * adjoint).sum()
        assert abs(lhs - rhs) / abs(lhs) < 1e-5, (lhs, rhs)

    def test_the_forward_maps_a_combined_image_to_coil_kspace(self) -> None:
        operator = MulticoilEIOperator(_maps(), _mask(), where="in the test")
        image = torch.randn(_B, 1, _S, _S)
        assert operator.forward(image).shape == (_B, _C, _S, _S)

    def test_the_model_input_is_two_real_channels(self) -> None:
        """`in_channels: 2` is unchanged: the coils never reach the network."""
        operator = MulticoilEIOperator(_maps(), _mask(), where="in the test")
        model_input = operator.model_input(_kspace())
        assert model_input.shape == (_B, 2, _S, _S)
        assert not torch.is_complex(model_input)

    def test_the_mask_is_applied(self) -> None:
        """An unmasked operator would invert a problem the arm is not solving."""
        maps, mask = _maps(), _mask()
        masked = MulticoilEIOperator(maps, mask, where="t").forward(torch.randn(_B, 1, _S, _S))
        assert torch.allclose(masked[..., 1::4, :], torch.zeros_like(masked[..., 1::4, :]))


class TestItRefusesWhatWouldSilentlyChangeTheOperator:
    def test_resolve_returns_none_when_the_knob_is_off(self) -> None:
        assert MulticoilEIOperator.resolve(False, {"coil_sensitivities": _maps()}, _mask()) is None

    def test_absent_maps_raise_rather_than_reverting(self) -> None:
        """Reverting to A = M F would report an EI result for a solved problem."""
        with pytest.raises(ValueError, match="requires complex coil"):
            MulticoilEIOperator.resolve(True, {}, _mask())

    def test_magnitude_maps_raise(self) -> None:
        """Coil phase is what makes the multi-coil operator better conditioned."""
        with pytest.raises(ValueError, match="COMPLEX"):
            MulticoilEIOperator.resolve(True, {"coil_sensitivities": _maps().abs()}, _mask())


class TestPrepareHandsOverACleanContext:
    def test_it_clears_use_dc(self) -> None:
        """The adjoint is already applied; the generic bridge would iFFT an image."""
        operator = MulticoilEIOperator(_maps(), _mask(), where="t")
        _, context = operator.prepare({"measured_kspace": _kspace(), "use_dc": True}, _kspace())
        assert context["use_dc"] is False

    def test_it_does_not_mutate_the_callers_context(self) -> None:
        operator = MulticoilEIOperator(_maps(), _mask(), where="t")
        original = {"measured_kspace": _kspace(), "use_dc": True}
        operator.prepare(original, _kspace())
        assert original["use_dc"] is True

    def test_it_prefers_the_measured_kspace_over_the_input(self) -> None:
        """`reconstruct` sets `measured_kspace` per branch; that is the truth."""
        operator = MulticoilEIOperator(_maps(), _mask(), where="t")
        measured = _kspace()
        from_context, _ = operator.prepare(
            {"measured_kspace": measured}, torch.zeros_like(measured)
        )
        assert torch.allclose(from_context, operator.model_input(measured))

    def test_it_falls_back_to_the_input_when_no_kspace_is_declared(self) -> None:
        operator = MulticoilEIOperator(_maps(), _mask(), where="t")
        supplied = _kspace()
        prepared, _ = operator.prepare({}, supplied)
        assert torch.allclose(prepared, operator.model_input(supplied))


class TestTheStrategyRoutesThroughIt:
    """Registering the operator is the easy half; the strategy must resolve it."""

    def test_the_strategy_reads_the_knob(self) -> None:
        import inspect

        from spectramr.infrastructure.training.strategies import equivariant_imaging_strategy

        source = inspect.getsource(equivariant_imaging_strategy)
        assert "self.multicoil_operator = bool(cfg.multicoil_operator)" in source

    def test_both_the_forward_and_the_adjoint_are_routed(self) -> None:
        """Forward only would leave validation inverting a different operator."""
        import inspect

        from spectramr.infrastructure.training.strategies.equivariant_imaging_strategy import (
            EquivariantImagingStrategy,
        )

        losses = inspect.getsource(EquivariantImagingStrategy._compute_losses_impl)
        inputs = inspect.getsource(EquivariantImagingStrategy._prepare_generator_inputs)
        assert "mc.forward" in losses, "the forward operator is not routed"
        assert "mc.prepare" in inputs, "the adjoint is not routed"

    def test_the_adjoint_override_covers_validation_too(self) -> None:
        """Overriding `_prepare_generator_inputs` is what makes that automatic."""
        from spectramr.infrastructure.training.strategies.equivariant_imaging_strategy import (
            EquivariantImagingStrategy,
        )

        assert "_prepare_generator_inputs" in vars(EquivariantImagingStrategy)

    def test_the_schema_defaults_to_the_combined_operator(self) -> None:
        """Every existing EI arm keeps byte-identical behaviour."""
        from spectramr.config.schemas.training.equivariant_imaging import (
            TrainingConfigEquivariantImaging,
        )

        assert TrainingConfigEquivariantImaging.model_fields["multicoil_operator"].default is False


class TestInterleavedToComplex:
    r"""The coercion that made ``A = M F`` transform the wrong quantity.

    The network emits a complex image real/imag-interleaved — that is what
    ``model.in_channels: 2`` means on an EI arm. The strategy used to coerce it
    with ``torch.complex(x, zeros_like(x))``, reading the two channels as two
    separate images with zero imaginary part.

    It broke both halves of the arm at once, and the cohort's first real run
    died on iteration 1:

        ValueError: transformed_recon shape (4, 2, 256, 256)
                 != prediction shape (8, 2, 256, 256)

    The batch doubled because ``_prepare_generator_inputs`` reads dim 1 of a
    complex 4-D tensor as SLICES and folds it into the batch — so a spurious
    second complex channel became a second slice.
    """

    def test_two_channels_become_one_complex_image(self) -> None:
        from spectramr.infrastructure.physics.multicoil_ei_operator import (
            interleaved_to_complex,
        )

        x = torch.randn(4, 2, 32, 32)
        out = interleaved_to_complex(x)
        assert out.shape == (4, 1, 32, 32)
        assert torch.is_complex(out)
        assert torch.allclose(out.real[:, 0], x[:, 0])
        assert torch.allclose(out.imag[:, 0], x[:, 1])

    def test_it_pairs_every_coil_not_just_the_first(self) -> None:
        from spectramr.infrastructure.physics.multicoil_ei_operator import (
            interleaved_to_complex,
        )

        x = torch.randn(2, 8, 16, 16)
        out = interleaved_to_complex(x)
        assert out.shape == (2, 4, 16, 16)
        assert torch.allclose(out.imag[:, 3], x[:, 7])

    def test_complex_passes_through(self) -> None:
        from spectramr.infrastructure.physics.multicoil_ei_operator import (
            interleaved_to_complex,
        )

        x = torch.randn(2, 3, 8, 8, dtype=torch.complex64)
        assert interleaved_to_complex(x) is x

    def test_an_odd_channel_count_raises(self) -> None:
        """Rather than silently zero-filling an imaginary part."""
        from spectramr.infrastructure.physics.multicoil_ei_operator import (
            interleaved_to_complex,
        )

        with pytest.raises(ValueError, match="interleaved reconstruction"):
            interleaved_to_complex(torch.randn(2, 3, 8, 8))

    def test_the_two_ei_branches_now_agree_on_batch(self) -> None:
        """The reported failure, as the shape identity that has to hold.

        Branch one is the reconstruction itself; branch two round-trips it
        through ``A`` and back through the k-space bridge. They are compared
        elementwise by ``EquivariantSSLReconLoss``, so their shapes must match.
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
        from spectramr.infrastructure.physics.multicoil_ei_operator import (
            interleaved_to_complex,
        )

        batch, size = 4, 64
        recon = torch.randn(batch, 2, size, size)
        mask = torch.zeros(1, 1, size, size)
        mask[..., ::4, :] = 1.0

        kspace = mask.to(torch.complex64) * fft2c(interleaved_to_complex(recon))
        # what `_prepare_generator_inputs` does with a complex 4-D k-space
        real_imag = torch.view_as_real(ifft2c(kspace))
        b, slices, h, w, _ = real_imag.shape
        model_input = real_imag.permute(0, 1, 4, 2, 3).contiguous().view(b * slices, 2, h, w)

        assert model_input.shape == recon.shape, (
            f"transformed {tuple(recon.shape)} != prediction {tuple(model_input.shape)}"
        )

    def test_the_forward_operator_sees_one_image_not_two(self) -> None:
        """The physics half: A must act on the complex image, not on its parts."""
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.infrastructure.physics.multicoil_ei_operator import (
            interleaved_to_complex,
        )

        recon = torch.randn(2, 2, 32, 32)
        fixed = fft2c(interleaved_to_complex(recon))
        assert fixed.shape == (2, 1, 32, 32)

        # the old coercion, kept as the contrast it is
        wrong = fft2c(torch.complex(recon, torch.zeros_like(recon)))
        assert wrong.shape == (2, 2, 32, 32)
        assert not torch.allclose(wrong[:, 0], fixed[:, 0])


class TestPrepareSuppressesBothAdjointRoutes:
    """``prepare`` has already run ``S^H F^H M``, so the generic builder must not
    transform again — and it now has TWO independent reasons to: ``use_dc`` and the
    declared k-space -> image bridge. Clearing one is no longer enough.
    """

    def test_it_stamps_both_suppression_keys(self) -> None:
        from spectramr.infrastructure.physics.multicoil_ei_operator import MulticoilEIOperator

        maps = torch.randn(1, 4, 8, 8, dtype=torch.complex64)
        mask = torch.ones(1, 1, 8, 8)
        op = MulticoilEIOperator(maps, mask, where="in the test")
        kspace = torch.randn(1, 4, 8, 8, dtype=torch.complex64)

        model_input, ctx = op.prepare({"measured_kspace": kspace}, kspace)

        assert ctx["use_dc"] is False
        assert ctx["kspace_adjoint_applied"] is True, (
            "the generic builder would ifft2c the SENSE-combined image a second time"
        )
        assert model_input.shape == (1, 2, 8, 8)
