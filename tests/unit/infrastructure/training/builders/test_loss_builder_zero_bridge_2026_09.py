"""The bridge guard is two-sided: two transforms is a defect, and so is none.

``#467`` added the double-bridge rejection -- a loss carrying its own
``DifferentiableFourierBridge`` declared under a list the builder also bridges,
so the tensor is inverse-transformed twice. The complement went unguarded: a
loss that expects an IMAGE, declared under a list the builder does NOT bridge,
which does not bridge itself either. Nothing transforms anything and it scores
raw k-space as anatomy.

Both shapes present identically -- finite values, no raise, a green run
measuring nothing it advertises -- which is why one-sidedness is not a
half-solved problem but a differently-shaped hole.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

_ARM = pathlib.Path(
    "experiments/inprogress/kspace_filling/attention_shootout/experiment_11_attention_none.yaml"
)

_KSPACE_ANCHOR = "  - name: complex_l1\n    weight: 1.0\n    enabled: true\n"


@pytest.fixture(scope="module")
def base_text() -> str:
    if not _ARM.exists():
        pytest.skip("kspace_filling cohort not present")
    return _ARM.read_text()


def _build(tmp_path: pathlib.Path, text: str) -> dict:
    path = tmp_path / "arm.yaml"
    path.write_text(text)
    cfg = TrainingSettings.from_yaml(str(path))
    return LossBuilder(cfg, torch.device("cpu")).build_reconstruction_losses().build()


def _under_kspace_losses(text: str, entry: str) -> str:
    assert _KSPACE_ANCHOR in text
    return text.replace(_KSPACE_ANCHOR, _KSPACE_ANCHOR + entry, 1)


class TestTheShippedArmStillBuilds:
    def test_the_barrier_bridges_through_complex_losses(self, base_text, tmp_path) -> None:
        """`complex_losses` adds `ifft_complex`, and the loss declares
        `input_domain: image` so it does not bridge itself. Exactly one."""
        assert "coil_subspace_residual" in _build(tmp_path, base_text)


class TestZeroBridgeIsRejected:
    def test_an_explicit_image_input_domain_under_kspace_losses(self, base_text, tmp_path) -> None:
        """Planted: the same loss moved to the unbridged list. Its own
        `use_fourier_bridge` is False because `input_domain: image`, so nothing
        transforms and it reads k-space as an image."""
        planted = _under_kspace_losses(
            base_text,
            "  - name: coil_subspace_residual\n"
            "    weight: 0.25\n"
            "    enabled: true\n"
            "    kwargs:\n"
            "      input_domain: image\n",
        )
        with pytest.raises(ConfigurationError, match="expects an IMAGE"):
            _build(tmp_path, planted)

    def test_a_registered_image_domain_loss_under_kspace_losses(self, base_text, tmp_path) -> None:
        """The second shape: no explicit kwarg, but the REGISTRY says image.
        `hfen` is the real such loss and the one #467's sibling check names."""
        planted = _under_kspace_losses(
            base_text, "  - name: hfen\n    weight: 0.3\n    enabled: true\n"
        )
        with pytest.raises(ConfigurationError, match="expects an IMAGE"):
            _build(tmp_path, planted)

    def test_the_message_names_both_remedies(self, base_text, tmp_path) -> None:
        """A guard that cannot be acted on gets suppressed rather than fixed."""
        planted = _under_kspace_losses(
            base_text, "  - name: hfen\n    weight: 0.3\n    enabled: true\n"
        )
        with pytest.raises(ConfigurationError) as excinfo:
            _build(tmp_path, planted)
        message = str(excinfo.value)
        assert "losses.image_losses" in message
        assert "input_domain: kspace" in message


class TestItDoesNotOverreach:
    def test_a_kspace_loss_under_kspace_losses_is_fine(self, base_text, tmp_path) -> None:
        """`null_space_content` is registered `kspace` and belongs exactly where
        it is; a guard that fired here would be worse than no guard."""
        built = _build(tmp_path, base_text)
        assert "null_space_content" in built

    def test_a_self_bridging_loss_under_kspace_losses_is_fine(self, base_text, tmp_path) -> None:
        """`sense_adjoint_l1` sets `use_fourier_bridge=True` and does the single
        iFFT itself -- one bridge, just not the builder's."""
        built = _build(tmp_path, base_text)
        assert "sense_adjoint_l1" in built
        assert getattr(built["sense_adjoint_l1"], "use_fourier_bridge", False) is True

    def test_the_double_bridge_guard_still_fires(self, base_text, tmp_path) -> None:
        """The sibling this complements must not have been traded away: the
        barrier under `complex_losses` with `input_domain: kspace` self-bridges
        AND gets the builder's bridge."""
        planted = base_text.replace(
            "      input_domain: image\n", "      input_domain: kspace\n", 1
        )
        with pytest.raises(ConfigurationError, match="inverse-transformed twice"):
            _build(tmp_path, planted)
