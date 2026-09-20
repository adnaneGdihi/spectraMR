"""The bf16-needs-Ampere gate.

bf16 below sm_80 is *emulated*: it runs, slowly, and numerically unlike what the
arm declared. ``torch.cuda.is_bf16_supported()`` reports that as support, so the
gate reads the compute capability instead. Capabilities are injected here, so
none of this needs a GPU.
"""

from __future__ import annotations

import pytest

from spectramr.core.device_capabilities import build_capabilities
from spectramr.infrastructure.training.mixed_precision import (
    AMPPrecisionUnsupportedError,
    MixedPrecisionConfig,
    MixedPrecisionIntegrationHelper,
    assert_amp_dtype_supported,
)

V100 = build_capabilities("cuda", (7, 0), triton=True, source="test")
TURING = build_capabilities("cuda", (7, 5), triton=True, source="test")
ADA = build_capabilities("cuda", (8, 9), triton=True, source="test")
UNKNOWN = build_capabilities("cuda", None, triton=True, source="unknown")
CPU = build_capabilities("cpu", None, triton=False, source="test")


class TestTheGate:
    @pytest.mark.parametrize("caps", [V100, TURING], ids=["sm_70", "sm_75"])
    def test_bf16_on_pre_ampere_raises(self, caps):
        with pytest.raises(AMPPrecisionUnsupportedError, match=r"no native bf16|Native bf16"):
            assert_amp_dtype_supported("bf16", capabilities=caps)

    def test_bf16_on_ampere_passes(self):
        assert_amp_dtype_supported("bf16", capabilities=ADA)

    def test_fp16_is_never_gated(self):
        """Only bf16 has the emulation problem."""
        for caps in (V100, TURING, ADA, UNKNOWN, CPU):
            assert_amp_dtype_supported("fp16", capabilities=caps)

    def test_a_non_cuda_device_is_out_of_scope(self):
        """CPU/MPS bf16 is the backlog's problem, not this gate's."""
        assert_amp_dtype_supported("bf16", capabilities=CPU)

    def test_an_unknown_capability_does_not_raise(self):
        """The audit legitimately runs where the compute node is not visible.
        'Cannot tell' must not be reported as 'cannot run'."""
        assert_amp_dtype_supported("bf16", capabilities=UNKNOWN)

    def test_an_unknown_capability_says_so(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            assert_amp_dtype_supported("bf16", capabilities=UNKNOWN)
        assert any("could not be read" in r.message for r in caplog.records)


class TestTheMessage:
    """The message has to pre-empt the obvious rebuttal."""

    def test_it_names_the_capability_and_the_alternatives(self):
        with pytest.raises(AMPPrecisionUnsupportedError) as excinfo:
            assert_amp_dtype_supported("bf16", capabilities=V100)
        message = str(excinfo.value)
        assert "7.0" in message
        assert "float16" in message and "float32" in message

    def test_it_pre_empts_the_is_bf16_supported_rebuttal(self):
        """Someone will check ``torch.cuda.is_bf16_supported()``, get True, and
        conclude the gate is broken. The message has to get there first."""
        with pytest.raises(AMPPrecisionUnsupportedError) as excinfo:
            assert_amp_dtype_supported("bf16", capabilities=V100)
        assert "is_bf16_supported" in str(excinfo.value)

    def test_it_does_not_offer_a_downgrade_as_automatic(self):
        """Raising, not downgrading: ``get_autocast_context`` turns fp16 into a
        nullcontext for complex arms, so an automatic downgrade would silently
        produce fp32 on exactly the arms that matter here."""
        with pytest.raises(AMPPrecisionUnsupportedError) as excinfo:
            assert_amp_dtype_supported("bf16", capabilities=V100)
        assert "falling back" not in str(excinfo.value).lower()


class TestTheHelperEnforcesIt:
    def test_constructing_with_bf16_on_a_v100_raises(self):
        with pytest.raises(AMPPrecisionUnsupportedError):
            MixedPrecisionIntegrationHelper(
                MixedPrecisionConfig(enabled=True, precision="bf16"),
                device="cuda",
                capabilities=V100,
            )

    def test_constructing_with_bf16_on_ampere_is_fine(self):
        helper = MixedPrecisionIntegrationHelper(
            MixedPrecisionConfig(enabled=True, precision="bf16"),
            device="cuda",
            capabilities=ADA,
        )
        assert helper.device_type == "cuda"

    def test_amp_disabled_is_not_gated(self):
        """A dtype declared under ``enabled: false`` never runs, so gating it
        would refuse a configuration that is merely inert."""
        helper = MixedPrecisionIntegrationHelper(
            MixedPrecisionConfig(enabled=False, precision="bf16"),
            device="cuda",
            capabilities=V100,
        )
        assert helper.enabled is False

    def test_fp16_on_a_v100_still_builds_its_scaler(self):
        """The gate must not disturb the path it does not govern."""
        helper = MixedPrecisionIntegrationHelper(
            MixedPrecisionConfig(enabled=True, precision="fp16"),
            device="cuda",
            capabilities=V100,
        )
        assert helper.scaler is not None
