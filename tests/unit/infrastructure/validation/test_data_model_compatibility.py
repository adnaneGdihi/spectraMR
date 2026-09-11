"""Unit tests for ``check_data_model_compatibility`` and
``check_model_loss_output_domain`` (Phases 2 + 5).

The tests exercise the audit behavior at the four corners of the
contract table: match, mismatch + bridging adapter, mismatch + no
adapter, and unannotated (legacy escape hatch).
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker
from spectramr.models.capabilities import ModelCapabilities
from spectramr.models.registry import MODEL_REGISTRY
from tests.utils.config_block_stub import block_stub
from tests.utils.data_config_stub import DataConfigStub


class _SmallNamespace:
    """Minimal stand-in for the Pydantic config object the audit reads.

    The audit functions reach into ``config.data.*``, ``config.model.*``,
    ``config.losses.*``, ``config.adapters.*``. A SimpleNamespace tree is
    enough — the audit uses defensive ``getattr`` everywhere.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_config(
    *,
    model_type: str,
    patch_size: tuple[int, ...],
    dataset_type: str = "nifti_paired",
    losses_output_domain: str | None = "image",
    adapters: object | None = None,
):
    """Build a minimal config object the audit can read."""
    # `data:` is decomposed -- patch_size/coil_processing_mode/normalization_type/
    # target_channels all live in sub-blocks now. DataConfigStub routes the flat
    # spellings to the homes the readers walk; unknown kwargs pass through.
    data = DataConfigStub(
        dataset_type=dataset_type,
        patch_size=patch_size,
        coil_processing_mode="none",
        normalization_type="minmax",
        target_channels=1,
        in_channels=1,
        voxel_mm=None,
        multi_contrast=_SmallNamespace(enabled=False),
    )
    model = _SmallNamespace(
        model_type=model_type,
        in_channels=1,
        out_channels=1,
        spatial_dims=len([d for d in patch_size if d > 1]),
        input_type="image",
    )
    # `losses.output_domain` -> `losses.policy.output_domain`; the *_losses
    # lists stayed top-level.
    losses = block_stub(
        "losses",
        output_domain=losses_output_domain,
        image_losses=[],
        kspace_losses=[],
        complex_losses=[],
    )
    return _SmallNamespace(
        data=data, model=model, losses=losses, adapters=adapters,
    )


def _annotate(model_type: str, **caps_kwargs):
    """Test helper: add capabilities to a registered model.

    Idempotent — replaces any existing capabilities for the test's
    duration. Pair with :func:`_unannotate` in teardown.
    """
    if model_type not in MODEL_REGISTRY:
        MODEL_REGISTRY[model_type] = {
            "class": object,
            "mode": "test",
            "capabilities": ModelCapabilities(**caps_kwargs),
        }
    else:
        MODEL_REGISTRY[model_type]["capabilities"] = ModelCapabilities(**caps_kwargs)


def _unannotate(model_type: str, *, was_test_only: bool = False):
    if was_test_only:
        MODEL_REGISTRY.pop(model_type, None)
    elif model_type in MODEL_REGISTRY:
        MODEL_REGISTRY[model_type]["capabilities"] = ModelCapabilities()


# ---------------------------------------------------------------------------
# check_data_model_compatibility
# ---------------------------------------------------------------------------


class TestDataModelCompatibility:
    def test_match_passes(self):
        _annotate("__test_match", spatial_dims=(3,), input_domain="image")
        try:
            cfg = _make_config(model_type="__test_match", patch_size=(64, 64, 64))
            r = ConfigHealthChecker().check_data_model_compatibility(cfg)
            assert r.passed and r.severity == "info"
        finally:
            _unannotate("__test_match", was_test_only=True)

    def test_mismatch_no_adapter_errors(self):
        _annotate("__test_mismatch", spatial_dims=(2,), input_domain="image")
        try:
            cfg = _make_config(model_type="__test_mismatch", patch_size=(64, 64, 64))
            r = ConfigHealthChecker().check_data_model_compatibility(cfg)
            assert not r.passed and r.severity == "error"
            assert "spatial_dims" in r.message
            assert "no silent rescue" in r.message.lower()
        finally:
            _unannotate("__test_mismatch", was_test_only=True)

    def test_mismatch_with_bridging_adapter_passes(self):
        _annotate("__test_bridged", spatial_dims=(2,), input_domain="image")
        try:
            adapter_cfg = _SmallNamespace(
                pre_model=[_SmallNamespace(name="slice_3d_to_2d", enabled=True, model_dump=lambda: {"name": "slice_3d_to_2d", "enabled": True})],
                post_model=[],
                pre_loss_pred=[],
                pre_loss_target=[],
                pre_metric=[],
            )
            cfg = _make_config(
                model_type="__test_bridged",
                patch_size=(64, 64, 64),
                adapters=adapter_cfg,
            )
            r = ConfigHealthChecker().check_data_model_compatibility(cfg)
            assert r.passed and r.severity == "info"
            assert "bridged" in r.message.lower()
        finally:
            _unannotate("__test_bridged", was_test_only=True)

    def test_unannotated_skipped(self):
        # A registered model without capabilities should skip the check.
        if "__test_unannot" in MODEL_REGISTRY:
            MODEL_REGISTRY.pop("__test_unannot")
        MODEL_REGISTRY["__test_unannot"] = {
            "class": object, "mode": "test",
            "capabilities": ModelCapabilities(),  # all None
        }
        try:
            cfg = _make_config(model_type="__test_unannot", patch_size=(64, 64, 64))
            r = ConfigHealthChecker().check_data_model_compatibility(cfg)
            assert r.passed and r.severity == "info"
            assert "unannotated" in r.message.lower() or "skip" in r.message.lower()
        finally:
            MODEL_REGISTRY.pop("__test_unannot", None)


# ---------------------------------------------------------------------------
# check_target_domain_matches_registered_output_domain (F6 / E-VIZ2)
# ---------------------------------------------------------------------------


class TestTargetDomainMatchesRegisteredOutputDomain:
    """``model.target_domain`` is Priority-1 in ``infer_output_domain`` and
    overrides the registered ``output_domain``; a mismatch forces a spurious IFFT
    on viz → k-space-rendered-as-image (E-VIZ2, smoke 2026-06-16, exp_p7). The
    check fires on annotated mismatches and abstains on unannotated models.

    Four corners: match, mismatch (the exp_p7 regression), unannotated (the
    exp_vf_27 escape hatch), and unset target_domain.
    """

    _CHECK = "check_target_domain_matches_registered_output_domain"

    def test_match_passes(self):
        _annotate("__td_match", output_domain="image")
        try:
            cfg = _make_config(model_type="__td_match", patch_size=(64, 64))
            cfg.model.target_domain = "image"
            r = getattr(ConfigHealthChecker(), self._CHECK)(cfg)
            assert r.passed and r.severity == "info"
        finally:
            _unannotate("__td_match", was_test_only=True)

    def test_mismatch_errors(self):
        # exp_p7: an image-output model declared ``target_domain: kspace``.
        _annotate("__td_mismatch", output_domain="image")
        try:
            cfg = _make_config(model_type="__td_mismatch", patch_size=(64, 64))
            cfg.model.target_domain = "kspace"
            r = getattr(ConfigHealthChecker(), self._CHECK)(cfg)
            assert not r.passed and r.severity == "error"
            assert "target_domain" in r.message
            assert "E-VIZ2" in r.message or "k-space" in r.message.lower()
            assert "model.target_domain" in r.yaml_keys
        finally:
            _unannotate("__td_mismatch", was_test_only=True)

    def test_unannotated_skipped(self):
        # exp_vf_27: reciprocity_divisor is unannotated → cannot cross-check.
        MODEL_REGISTRY.pop("__td_unannot", None)
        MODEL_REGISTRY["__td_unannot"] = {
            "class": object,
            "mode": "test",
            "capabilities": ModelCapabilities(),  # output_domain None
        }
        try:
            cfg = _make_config(model_type="__td_unannot", patch_size=(64, 64))
            cfg.model.target_domain = "kspace"
            r = getattr(ConfigHealthChecker(), self._CHECK)(cfg)
            assert r.passed and r.severity == "info"
            assert "unannotated" in r.message.lower() or "skip" in r.message.lower()
        finally:
            MODEL_REGISTRY.pop("__td_unannot", None)

    def test_unset_target_domain_skipped(self):
        _annotate("__td_unset", output_domain="image")
        try:
            cfg = _make_config(model_type="__td_unset", patch_size=(64, 64))
            # no target_domain attribute at all → skip
            r = getattr(ConfigHealthChecker(), self._CHECK)(cfg)
            assert r.passed and r.severity == "info"
            assert "unset" in r.message.lower() or "skip" in r.message.lower()
        finally:
            _unannotate("__td_unset", was_test_only=True)


# ---------------------------------------------------------------------------
# 1-D sequence / navigator encoder exemption (smoke_audit_20260526 / ib_infonce)
# ---------------------------------------------------------------------------


class TestOneDNavigatorExemption:
    """A model registered ``spatial_dims=(1,)`` (e.g. ``nav_encoder_1d`` in the
    IB-VF arm) consumes a 1-D signal the *strategy* extracts from the 2-D
    acquisition (the DC k-space navigator m(t)=y(0,t)), NOT the coil image. The
    generic coil-channel / 2-D-spatial cross-checks must NOT fire on it."""

    def test_data_model_compat_skips_1d_navigator(self):
        _annotate("__test_nav1d", spatial_dims=(1,), input_domain="kspace")
        try:
            cfg = _make_config(
                model_type="__test_nav1d", patch_size=(256, 256, 1),
                dataset_type="kspace",
            )
            r = ConfigHealthChecker().check_data_model_compatibility(cfg)
            assert r.passed and r.severity == "info", r.message
            assert "navigator" in r.message.lower() or "1-d" in r.message.lower()
        finally:
            _unannotate("__test_nav1d", was_test_only=True)

    def test_domain_alignment_skips_1d_navigator(self):
        cfg = _make_config(
            model_type="nav_encoder_1d", patch_size=(256, 256, 1),
            dataset_type="kspace",
        )
        # IB-VF navigator encoder: 1-D model, svd coils, 2-ch navigator in,
        # 64-d latent out — coil-channel math (svd×4=8) must NOT fire.
        cfg.model.spatial_dims = 1
        cfg.model.in_channels = 2
        cfg.model.out_channels = 64
        # The real sub-block schemas are frozen=True, exactly like a live
        # config; rebuild the block rather than assigning through it.
        cfg.data.coils = cfg.data.coils.model_copy(
            update={"processing_mode": "svd", "num_virtual_coils": 4}
        )
        results = ConfigHealthChecker().check_domain_alignment(cfg)
        errors = [r for r in results if not r.passed]
        assert not errors, (
            "domain_alignment must skip the coil-channel check for a 1-D "
            f"navigator encoder: {[e.message for e in errors]}"
        )

    def test_domain_alignment_still_flags_2d_channel_mismatch(self):
        """Guard the gate is not over-broad: a genuine 2-D model with a coil
        channel mismatch must still fail."""
        cfg = _make_config(
            model_type="some_2d_recon", patch_size=(256, 256, 1),
            dataset_type="kspace",
        )
        cfg.model.spatial_dims = 2
        cfg.model.in_channels = 2  # mismatches svd×4 = 8
        cfg.model.out_channels = 2
        # The real sub-block schemas are frozen=True, exactly like a live
        # config; rebuild the block rather than assigning through it.
        cfg.data.coils = cfg.data.coils.model_copy(
            update={"processing_mode": "svd", "num_virtual_coils": 4}
        )
        results = ConfigHealthChecker().check_domain_alignment(cfg)
        assert any(not r.passed for r in results), (
            "2-D model with in_channels=2 vs svd×4=8 must still fail domain_alignment"
        )


# ---------------------------------------------------------------------------
# check_model_loss_output_domain
# ---------------------------------------------------------------------------


class TestModelLossOutputDomain:
    def test_exact_match_passes(self):
        _annotate("__test_oo_match", output_domain="image")
        try:
            cfg = _make_config(
                model_type="__test_oo_match", patch_size=(64, 64, 1),
                losses_output_domain="image",
            )
            r = ConfigHealthChecker().check_model_loss_output_domain(cfg)
            assert r.passed and "matches" in r.message
        finally:
            _unannotate("__test_oo_match", was_test_only=True)

    @pytest.mark.parametrize(
        "model_out,loss_out",
        [
            ("image", "kspace"),         # LossBuilder auto-FFT
            ("image", "complex_image"),  # LossBuilder auto-cast
            ("pde_grid", "image"),       # Tensor-shape equivalent
            ("image", "pde_grid"),
        ],
    )
    def test_known_bridges_pass(self, model_out, loss_out):
        _annotate("__test_oo_bridge", output_domain=model_out)
        try:
            cfg = _make_config(
                model_type="__test_oo_bridge", patch_size=(64, 64, 1),
                losses_output_domain=loss_out,
            )
            r = ConfigHealthChecker().check_model_loss_output_domain(cfg)
            assert r.passed
            assert "bridged" in r.message.lower() or "matches" in r.message.lower()
        finally:
            _unannotate("__test_oo_bridge", was_test_only=True)

    def test_unbridged_direction_errors(self):
        # kspace → image is NOT auto-bridged (LossBuilder only does the
        # other direction). Audit must flag.
        _annotate("__test_oo_unbridged", output_domain="kspace")
        try:
            cfg = _make_config(
                model_type="__test_oo_unbridged", patch_size=(64, 64, 1),
                losses_output_domain="image",
            )
            r = ConfigHealthChecker().check_model_loss_output_domain(cfg)
            assert not r.passed and r.severity == "error"
            assert "kspace" in r.message and "image" in r.message
            assert r.fix_hint and "pre_loss_pred" in r.fix_hint
        finally:
            _unannotate("__test_oo_unbridged", was_test_only=True)

    def test_unannotated_skipped(self):
        MODEL_REGISTRY.pop("__test_oo_unannot", None)
        MODEL_REGISTRY["__test_oo_unannot"] = {
            "class": object, "mode": "test",
            "capabilities": ModelCapabilities(),
        }
        try:
            cfg = _make_config(
                model_type="__test_oo_unannot", patch_size=(64, 64, 1),
                losses_output_domain="image",
            )
            r = ConfigHealthChecker().check_model_loss_output_domain(cfg)
            assert r.passed and r.severity == "info"
            assert "unannotated" in r.message.lower()
        finally:
            MODEL_REGISTRY.pop("__test_oo_unannot", None)

    def test_losses_output_domain_unset_skipped(self):
        _annotate("__test_oo_no_loss", output_domain="image")
        try:
            cfg = _make_config(
                model_type="__test_oo_no_loss", patch_size=(64, 64, 1),
                losses_output_domain=None,
            )
            r = ConfigHealthChecker().check_model_loss_output_domain(cfg)
            assert r.passed and r.severity == "info"
            assert "unset" in r.message.lower()
        finally:
            _unannotate("__test_oo_no_loss", was_test_only=True)
