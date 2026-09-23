"""The k-space -> image adjoint is its own decision, not a side effect of ``use_dc``.

``_prepare_generator_inputs`` applied ``ifft2c`` to the model input ``if use_dc``,
and ``use_dc`` is ``strategy.dc_layer is not None``.
``StrategyInitializationHelper.initialize_data_consistency`` clears ``dc_layer``
whenever the generator carries no built-in DC module — with a log *warning*, not a
raise — so an arm declaring ``physics.data_consistency.enabled: true`` over a plain
U-Net got **neither** data consistency **nor** the adjoint, and the network was fed
raw k-space while the loss, the metrics and the previewer all read its output as an
image.

Observed on ``ei_unet_m4raw_r4`` and ``ei_control_mc_only_m4raw_r4`` (cluster run
2026-09-20, ``45e4af988ba6``): four ``🧲 Strategy DC: Enabled in config but NOT found
in generator model`` warnings, ``input_raw == input_prepared`` at ``min=-35.25
max=42.94`` (k-space dynamic range) in every ``first_steps`` snapshot, and a
``mri_a7_kspace_recon`` "proposed" panel that is the DC spike — the white blob at the
centre of an otherwise black frame.

Each test below plants one shape the rule has to cover (non-negotiable 15): the
missing bridge, the bridge that must NOT fire on a k-space-native model, the strategy
that already applied the adjoint itself, and the config that lies about its domain.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.utils.transform_ops import FFTTransformer
from spectramr.models.blocks.attention_domains import complex_to_interleaved


class _Plain(nn.Module):
    """A generator with no ``dc_layer``, no ``encoder`` and no extra forward kwargs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - never called
        return x


@pytest.fixture(autouse=True, scope="module")
def _registry() -> None:
    """``get_model_capabilities`` answers ``None`` on an unpopulated registry, so
    every bridge decision would silently be "no" (→ ``registries.md``)."""
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()


def _strategy(model_type: str, *, dataset_type: str = "kspace") -> ReconstructionTrainingStrategy:
    """A bare strategy carrying only what ``_prepare_generator_inputs`` reads.

    ``__new__`` rather than ``__init__``: constructing the real thing needs a built
    model, an optimizer and a DI container, none of which this seam touches.
    """
    s = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    s.config = SimpleNamespace(
        model=SimpleNamespace(model_type=model_type, in_channels=2),
        data=SimpleNamespace(
            dataset_type=dataset_type,
            multislice_enabled=False,
            processing=SimpleNamespace(enable_kspace_normalization=True),
            coils=SimpleNamespace(processing_mode="none"),
        ),
    )
    s.env = SimpleNamespace(generator=_Plain(), config=s.config)
    s.fft_transformer = FFTTransformer(device=torch.device("cpu"))
    s.logging_service = None
    return s


def _kspace_batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``(image, kspace)`` for a 2-channel real-stacked single virtual coil."""
    g = torch.Generator().manual_seed(seed)
    # A centred blob, so the k-space carries the DC signature the guard looks for.
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    blob = torch.exp(-((yy - 16.0) ** 2 + (xx - 16.0) ** 2) / 40.0)
    image = (blob + 0.02 * torch.rand(32, 32, generator=g)).to(torch.complex64)[None, None]
    kspace = fft2c(image)
    real_stacked = torch.cat([kspace.real, kspace.imag], dim=1)
    return image, real_stacked


class TestTheBridgeFiresWithoutADataConsistencyLayer:
    """The planted violation: this is red on the pre-fix code."""

    def test_an_image_domain_model_over_a_kspace_loader_receives_the_adjoint(self) -> None:
        """``standard_unet`` declares ``input_domain='image'``; the loader serves k-space."""
        s = _strategy("standard_unet")
        image, kspace = _kspace_batch()
        ctx = {"use_dc": False, "measured_kspace": kspace, "mask": None}

        lr_image, _ = s._prepare_generator_inputs(ctx, kspace)

        expected = complex_to_interleaved(ifft2c(kspace))
        assert lr_image.shape == (1, 2, 32, 32), (
            "the model input is still raw k-space — the network is being asked to "
            "learn a global Fourier transform with local convolutions, and its "
            "output renders as the DC-spike blob"
        )
        assert torch.allclose(lr_image, expected, atol=1e-5)
        # And the adjoint really did land back on the image it came from, in the
        # (R0, I0) layout `standard_unet`'s in_channels: 2 names.
        assert torch.allclose(lr_image, complex_to_interleaved(image), atol=1e-5)

    def test_the_decision_is_cached_rather_than_re_resolved_every_step(self) -> None:
        """It reads the registry and the config, neither of which moves in a run
        (non-negotiable 9: nothing per-step that does not have to be)."""
        s = _strategy("standard_unet")
        _, kspace = _kspace_batch()
        assert s._kspace_bridge is None
        s._prepare_generator_inputs({"use_dc": False, "measured_kspace": kspace}, kspace)
        assert s._kspace_bridge is True
        s._prepare_generator_inputs({"use_dc": False, "measured_kspace": kspace}, kspace)
        assert s._kspace_bridge_checked is True


class TestTheBridgeStaysOffWhereItWouldBeWrong:
    """Over-reach is the other failure: an IFFT the arm never asked for."""

    def test_a_kspace_native_model_keeps_its_kspace_input(self) -> None:
        """``complex_unet`` declares ``input_domain='kspace'`` — the n2n cohort's
        backbone. Bridging it would hand it an image and silently change what all
        five ``n2n_*`` arms train on."""
        s = _strategy("complex_unet")
        _, kspace = _kspace_batch()

        lr_image, _ = s._prepare_generator_inputs(
            {"use_dc": False, "measured_kspace": kspace}, kspace
        )

        assert torch.equal(lr_image, kspace)
        assert s._kspace_bridge is False

    def test_an_unannotated_model_is_left_alone(self) -> None:
        """443 of 593 registered models declare no ``input_domain``. "Nobody said"
        must not read as "image" — that guess is what this function exists to end."""
        s = _strategy("varnet")
        _, kspace = _kspace_batch()
        lr_image, _ = s._prepare_generator_inputs(
            {"use_dc": False, "measured_kspace": kspace}, kspace
        )
        assert torch.equal(lr_image, kspace)

    def test_an_image_loader_does_not_trigger_the_bridge(self) -> None:
        """No k-space anywhere: an image-domain model over an image-domain loader."""
        s = _strategy("standard_unet", dataset_type="nifti_paired")
        s.config.data.processing.enable_kspace_normalization = False
        image = torch.rand(1, 2, 32, 32)
        lr_image, _ = s._prepare_generator_inputs({"use_dc": False, "measured_kspace": None}, image)
        assert torch.equal(lr_image, image)

    def test_a_strategy_that_already_applied_the_adjoint_suppresses_it(self) -> None:
        """``MulticoilEIOperator.prepare`` runs ``S^H F^H M`` itself. Clearing
        ``use_dc`` no longer suppresses the bridge on its own, so it stamps
        ``kspace_adjoint_applied`` — without which the multi-coil EI arm would
        IFFT its own reconstruction."""
        s = _strategy("standard_unet")
        _, kspace = _kspace_batch()
        already_image = torch.rand(1, 2, 32, 32)

        lr_image, _ = s._prepare_generator_inputs(
            {
                "use_dc": False,
                "kspace_adjoint_applied": True,
                "measured_kspace": kspace,
            },
            already_image,
        )

        assert torch.equal(lr_image, already_image)


class TestTheDeclaredDomainIsCheckedAgainstTheTensor:
    """A config that declares a k-space loader while the loader serves images would
    otherwise train on a Hermitian-symmetric 'doubled brain' and report success."""

    def test_an_image_tensor_under_a_declared_kspace_loader_raises(self) -> None:
        s = _strategy("standard_unet")
        image = torch.rand(1, 2, 32, 32)  # no DC spike: this is not k-space
        with pytest.raises(ValueError, match="no k-space DC signature"):
            s._prepare_generator_inputs({"use_dc": False, "measured_kspace": image}, image)

    def test_the_check_runs_once_and_not_per_step(self) -> None:
        """``looks_like_kspace`` needs a host sync, which is forbidden inside the
        training loop (non-negotiable 9)."""
        s = _strategy("standard_unet")
        _, kspace = _kspace_batch()
        s._prepare_generator_inputs({"use_dc": False, "measured_kspace": kspace}, kspace)
        assert s._kspace_bridge_checked is True

        # A later image-domain batch is NOT re-probed: the question was settled.
        image = torch.rand(1, 2, 32, 32)
        s._prepare_generator_inputs({"use_dc": False, "measured_kspace": image}, image)
