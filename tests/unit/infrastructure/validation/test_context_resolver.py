"""Unit tests for the context resolver (cached-cascade WS-X Phase-0 freeze).

These pin the FROZEN signature ``resolve(config) -> ResolvedExperimentContext``
and the fail-soft contract (never crash on a partial config). Deep
data/physics/loss resolution + reconciliation-relocation is WS-X Phase A.
"""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.domain.entities import ResolvedExperimentContext
from spectramr.infrastructure.validation.context_resolver import resolve
from spectramr.models.capabilities import ModelCapabilities, StrategyCapabilities


def _fake_config(**over: object) -> SimpleNamespace:
    base = SimpleNamespace(
        # `run.config_version`, matching the real schema since the phase-5 `run:`
        # rename. This stand-in kept the pre-rename ROOT spelling, so it agreed
        # with itself while the resolver -- reading the real object -- got
        # nothing and stamped "?" for months. A stand-in that models the old
        # shape does not test the resolver, it tests the stand-in;
        # `test_the_version_stamp_is_real_on_a_real_settings_object` below
        # drives the actual loader so the two cannot diverge again.
        run=SimpleNamespace(config_version="1.0"),
        model=SimpleNamespace(model_type="__fake_model__", target_domain="image"),
        training=SimpleNamespace(training_mode="reconstruction"),
        losses=None,
        data=None,
        physics=None,
    )
    for k, v in over.items():
        setattr(base, k, v)
    return base


def test_resolve_returns_frozen_ir() -> None:
    ctx = resolve(_fake_config())
    assert isinstance(ctx, ResolvedExperimentContext)
    assert ctx.config_version == "1.0"
    assert ctx.model_type == "__fake_model__"


def test_the_version_stamp_is_real_on_a_real_settings_object() -> None:
    """Drive the ACTUAL loader, not a stand-in.

    ``_get`` fails soft by design, so a wrong path yields the ``"?"`` default
    rather than an error — and the only other test touching this asserts ``"?"``
    for a deliberately empty config, so a genuinely broken read looked exactly
    like the fail-soft contract working. Reading a real ``TrainingSettings`` is
    the only assertion that can tell those two apart.
    """
    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
    from spectramr.config.settings import TrainingSettings

    settings = TrainingSettings.settings_from_dict(
        {
            "config_version": CANONICAL_CONFIG_VERSION,
            "data": {"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
            "model": {"model_type": "unet"},
        }
    )
    ctx = resolve(settings)
    assert ctx.config_version == CANONICAL_CONFIG_VERSION
    assert ctx.config_version != "?", "the resolver is reading the wrong path again"


def test_resolve_populates_adapter_chain_from_config() -> None:
    # Regression: an empty adapter_chain blinded the domain-chain rule to
    # config-declared bridges, false-rejecting correctly-bridged arms. The
    # resolver must look declared adapters up in the registry.
    cfg = _fake_config(
        adapters=SimpleNamespace(
            pre_loss_target=[SimpleNamespace(name="ifft_kspace_to_image")],
            pre_model=[],
            post_model=[],
            pre_metric=[],
        )
    )
    ctx = resolve(cfg)
    assert len(ctx.adapter_chain) == 1
    ad = ctx.adapter_chain[0]
    assert ad.bridges_from.get("domain") == "kspace"
    assert ad.bridges_to.get("domain") == "complex_image"


def test_resolve_adapter_chain_empty_when_no_adapters() -> None:
    assert resolve(_fake_config()).adapter_chain == ()


def test_unknown_model_is_failsoft_default_caps() -> None:
    # An unknown model name must not crash the resolver — it yields the empty
    # "unannotated" ModelCapabilities (skip-the-check convention).
    ctx = resolve(_fake_config())
    assert ctx.model_profile == ModelCapabilities()


def test_strategy_unresolvable_is_failsoft() -> None:
    # A duck-typed config the strategy factory cannot dispatch must fall back to
    # a non-crashing strategy name + empty capabilities, not raise.
    ctx = resolve(_fake_config())
    assert isinstance(ctx.strategy_profile, StrategyCapabilities)
    assert isinstance(ctx.strategy_name, str) and ctx.strategy_name


def test_resolve_never_raises_on_sparse_config() -> None:
    sparse = SimpleNamespace()  # no attributes at all
    ctx = resolve(sparse)
    assert isinstance(ctx, ResolvedExperimentContext)
    assert ctx.model_type == "<unknown>"
    assert ctx.config_version == "?"


def test_data_domain_is_what_the_loader_emits() -> None:
    """Dataset family first, coil-processing mode second; never the model's
    declared target domain (a proxy for the model OUTPUT, retired 2026-09-03)."""
    assert resolve(_fake_config()).data_profile.domain == "image"
    kspace = _fake_config(
        data=SimpleNamespace(dataset_type="m4raw", coils=SimpleNamespace(processing_mode="rss"))
    )
    assert resolve(kspace).data_profile.domain == "kspace"
    magnitude = _fake_config(
        data=SimpleNamespace(
            dataset_type="m4raw", coils=SimpleNamespace(processing_mode="rss_image")
        ),
        model=SimpleNamespace(model_type="__fake_model__", target_domain="kspace"),
    )
    assert resolve(magnitude).data_profile.domain == "image"


def test_resolve_spatial_dims_2d_from_patch_size() -> None:
    cfg = _fake_config(data=SimpleNamespace(patch_size=(256, 256, 1)))
    assert resolve(cfg).data_profile.spatial_dims == (2,)


def test_resolve_spatial_dims_3d_from_patch_size() -> None:
    cfg = _fake_config(data=SimpleNamespace(patch_size=(160, 160, 160)))
    assert resolve(cfg).data_profile.spatial_dims == (3,)


def test_resolve_spatial_dims_none_without_patch_size() -> None:
    assert resolve(_fake_config()).data_profile.spatial_dims is None


def test_resolve_spatial_dims_1d_from_patch_size() -> None:
    # MRF/sequence data: [N, 1, 1] is 1-D, not a 2-D image (regression for the
    # idea_3_conformal_fp_embedding false-reject).
    cfg = _fake_config(data=SimpleNamespace(patch_size=(1000, 1, 1)))
    assert resolve(cfg).data_profile.spatial_dims == (1,)


def test_resolve_spatial_dims_from_the_canonical_sampling_block() -> None:
    """Phase 9e moved `patch_size` under `data.sampling`."""
    cfg = _fake_config(data=SimpleNamespace(sampling=SimpleNamespace(patch_size=(256, 256, 1))))
    assert resolve(cfg).data_profile.spatial_dims == (2,)


def test_the_spatial_rank_is_real_on_a_real_settings_object() -> None:
    """Drive the ACTUAL loader, for the same reason as the version stamp above.

    The four stand-in tests here all set a FLAT ``patch_size`` on a
    ``SimpleNamespace``, which has whatever attribute you give it. They kept
    passing after phase 9e while the resolver -- reading a real
    ``DataConfigSchema`` -- got ``None`` on every arm in the corpus, so
    ``rule_spatial_rank`` (3-D model fed 2-D patches) silently returned ``[]``.
    A stand-in cannot tell "the guard is clean" from "the guard is blind";
    only a resolved settings object can.
    """
    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
    from spectramr.config.settings import TrainingSettings

    settings = TrainingSettings.settings_from_dict(
        {
            "config_version": CANONICAL_CONFIG_VERSION,
            "data": {
                "train_path": "/tmp/t",
                "val_path": "/tmp/v",
                "batch_size": 2,
                "patch_size": (256, 256, 1),
            },
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
            "model": {"model_type": "unet"},
        }
    )
    # The fold put it on the canonical path; the resolver must find it there.
    assert settings.data.sampling.patch_size == (256, 256, 1)
    assert resolve(settings).data_profile.spatial_dims == (2,)
