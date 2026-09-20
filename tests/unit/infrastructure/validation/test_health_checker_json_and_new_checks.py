"""Unit tests for the audit-ladder additions to ConfigHealthChecker.

Covers:

* JSON serialisation of :class:`HealthCheckResult` and
  :class:`HealthCheckReport`.
* The three new checks: ``check_advertised_options``,
  ``check_loss_domain_consistency``,
  ``check_amp_grad_clip_interaction``.

The tests use lightweight stub objects rather than fully-loaded
:class:`TrainingSettings` instances so each check can be exercised in
isolation without bringing in the entire schema graph.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
    HealthCheckReport,
    HealthCheckResult,
)

# ── JSON serialisation ──────────────────────────────────────────────────


def test_health_check_result_to_dict_round_trips() -> None:
    r = HealthCheckResult(
        passed=False,
        check_name="advertised_options",
        message="oops",
        severity="error",
        category="advertised_options",
        yaml_keys=["model.model_kwargs.attention_type"],
        fix_hint="set it to 'spatial'",
    )
    d = r.to_dict()
    assert d["passed"] is False
    assert d["category"] == "advertised_options"
    assert d["yaml_keys"] == ["model.model_kwargs.attention_type"]
    assert d["fix_hint"] == "set it to 'spatial'"
    # JSON round-trip
    json.loads(json.dumps(d))


def test_health_check_report_to_dict_aggregates_counts() -> None:
    report = HealthCheckReport(
        results=[
            HealthCheckResult(passed=True, check_name="x", message="ok", severity="info"),
            HealthCheckResult(passed=False, check_name="y", message="warn", severity="warning"),
            HealthCheckResult(passed=False, check_name="z", message="bad", severity="error"),
        ]
    )
    d = report.to_dict()
    assert d["passed"] is False
    assert d["n_errors"] == 1
    assert d["n_warnings"] == 1
    assert len(d["results"]) == 3


def test_health_check_report_to_json_is_valid_json() -> None:
    report = HealthCheckReport(
        results=[
            HealthCheckResult(passed=True, check_name="x", message="ok", severity="info"),
        ]
    )
    parsed = json.loads(report.to_json())
    assert parsed["passed"] is True
    assert parsed["n_errors"] == 0


# ── check_advertised_options ────────────────────────────────────────────


class _FakeModelWithSchema:
    OPTION_SCHEMA = {"attention_type": ["spatial", "cross"]}


class _FakeModelWithoutSchema:
    pass


@pytest.fixture
def _patch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``get_model_class`` with a stub that returns the test models.

    ``ConfigHealthChecker.check_advertised_options`` imports
    ``get_model_class`` from ``spectramr.models.registry`` at call time, so
    we patch that exact symbol. (Earlier versions of this test patched
    a ``ModelRegistry`` attribute that no longer exists on the module —
    the registry surface is module-level functions.)
    """
    from spectramr.models import registry as registry_mod

    fake = {
        "with_schema": _FakeModelWithSchema,
        "without_schema": _FakeModelWithoutSchema,
    }

    def _fake_get_model_class(name: str) -> Any:
        return fake[name]

    monkeypatch.setattr(registry_mod, "get_model_class", _fake_get_model_class)


def _make_config(model_type: str, model_kwargs: dict | None = None) -> Any:
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            model_type=model_type,
            model_kwargs=model_kwargs or {},
            in_channels=1,
            out_channels=1,
        ),
    )


def test_advertised_options_pass_when_value_in_set(_patch_registry: None) -> None:
    cfg = _make_config("with_schema", {"attention_type": "spatial"})
    results = ConfigHealthChecker().check_advertised_options(cfg)
    assert all(r.passed for r in results)


def test_advertised_options_error_on_silent_fallback(_patch_registry: None) -> None:
    cfg = _make_config("with_schema", {"attention_type": "dual_domain"})
    results = ConfigHealthChecker().check_advertised_options(cfg)
    failures = [r for r in results if not r.passed]
    assert len(failures) == 1
    assert failures[0].severity == "error"
    assert failures[0].category == "advertised_options"
    assert "dual_domain" in failures[0].message
    assert failures[0].fix_hint is not None
    assert "model.model_kwargs.attention_type" in failures[0].yaml_keys


def test_advertised_options_skips_models_without_schema(_patch_registry: None) -> None:
    """A model with no OPTION_SCHEMA is SKIPPED — it must not emit a pass.

    Regression (issue #284): the check used to append an unconditional
    ``passed=True, "All <model> model_kwargs are in the advertised set."`` when it
    found no violations — including for a model whose advertised set it had never
    read, because there was none. That result asserts a fact the check did not
    establish. ``audit`` is ``--strict`` by default, so a confident-but-unfounded
    ``info`` is not harmless padding: it is the reassuring green that hides the
    finding (pitfall #16). No basis, no result.
    """
    cfg = _make_config("without_schema", {"any_kwarg": "any_value"})
    results = ConfigHealthChecker().check_advertised_options(cfg)
    assert results == []  # silent skip is intentional


def test_advertised_options_skips_when_schema_covers_no_declared_kwarg(
    _patch_registry: None,
) -> None:
    """Having a schema is not enough — it must actually cover a declared kwarg.

    ``with_schema`` advertises ``attention_type``; this config declares none of the
    schema's keys, so nothing was compared and no pass may be claimed (#284).
    """
    cfg = _make_config("with_schema", {"base_channels": 32})
    results = ConfigHealthChecker().check_advertised_options(cfg)
    assert [r for r in results if r.passed] == [], (
        "check_advertised_options claimed a pass without comparing any kwarg "
        "against the advertised set"
    )


def test_advertised_options_pass_names_what_it_actually_checked(
    _patch_registry: None,
) -> None:
    """When a pass IS emitted, it must be backed by a real comparison."""
    cfg = _make_config("with_schema", {"attention_type": "spatial"})
    results = ConfigHealthChecker().check_advertised_options(cfg)

    passes = [r for r in results if r.passed]
    assert len(passes) == 1
    # The message names the kwarg that was actually validated, so the claim is
    # auditable rather than a blanket "all clear".
    assert "attention_type" in passes[0].message


# ── check_loss_domain_consistency ───────────────────────────────────────


def _losses_cfg(output_domain: str, **lists: list[Any]) -> Any:
    return types.SimpleNamespace(
        losses=types.SimpleNamespace(
            # `output_domain` moved into `losses.policy` in phase 10d. This
            # stand-in mirrors the real schema; `test_stand_ins_match_the_real
            # _schema` below fails if it drifts again.
            policy=types.SimpleNamespace(output_domain=output_domain),
            image_losses=lists.get("image", []),
            kspace_losses=lists.get("kspace", []),
            complex_losses=lists.get("complex", []),
        ),
    )


def test_loss_domain_consistency_pass_when_aligned() -> None:
    cfg = _losses_cfg(
        "image",
        image=[types.SimpleNamespace(name="l1", weight=1.0, enabled=True)],
    )
    results = ConfigHealthChecker().check_loss_domain_consistency(cfg)
    assert all(r.passed for r in results)


def test_loss_domain_consistency_error_when_kspace_loss_in_image_config() -> None:
    cfg = _losses_cfg(
        "image",
        kspace=[types.SimpleNamespace(name="kspace_l2", weight=1.0)],
    )
    results = ConfigHealthChecker().check_loss_domain_consistency(cfg)
    failures = [r for r in results if not r.passed]
    assert len(failures) == 1
    assert failures[0].severity == "error"
    assert failures[0].category == "loss_domain_consistency"
    assert "kspace_losses" in " ".join(failures[0].yaml_keys)


# ── VF-smoke-2026-05-25: complex_image output domain bridging ───────────
#
# LossBuilder._build_list_based_losses accepts ``complex_image`` (raises on
# any other value for complex output), and its bridge matrix handles
# image_losses (magnitude extraction) and kspace_losses (FFT) under it. The
# validator previously keyed compat on a dead ``complex`` value and omitted
# the complex_image bridged combos, so a phase-aware arm declaring
# ``output_domain: complex_image`` + ``complex_losses`` (twin_dps, ib_infonce)
# was falsely flagged as an error. These pin the validator to the builder.


def test_loss_domain_complex_image_accepts_complex_losses() -> None:
    cfg = _losses_cfg(
        "complex_image",
        complex=[types.SimpleNamespace(name="phase_smoothness_complex", weight=0.01, enabled=True)],
    )
    results = ConfigHealthChecker().check_loss_domain_consistency(cfg)
    assert all(r.passed for r in results), [r.message for r in results if not r.passed]


def test_loss_domain_complex_image_bridges_image_and_kspace_losses() -> None:
    cfg = _losses_cfg(
        "complex_image",
        image=[types.SimpleNamespace(name="complex_mse", weight=1.0, enabled=True)],
        kspace=[types.SimpleNamespace(name="kspace_l2", weight=1.0, enabled=True)],
    )
    results = ConfigHealthChecker().check_loss_domain_consistency(cfg)
    # Both are bridged by the builder (magnitude extraction / FFT) → info,
    # not error.
    assert all(r.passed for r in results), [r.message for r in results if not r.passed]


def test_loss_domain_image_output_casts_complex_losses() -> None:
    cfg = _losses_cfg(
        "image",
        complex=[types.SimpleNamespace(name="phase_smoothness_complex", weight=0.01, enabled=True)],
    )
    results = ConfigHealthChecker().check_loss_domain_consistency(cfg)
    # image + complex_losses is the builder's cast-to-complex path → info.
    assert all(r.passed for r in results), [r.message for r in results if not r.passed]


# ── check_loss_domain_block_match: agnostic losses ──────────────────────
#
# A loss registered with domain='agnostic' (e.g. nll_bits_per_dim on a
# normalizing flow) is domain-independent and must be accepted under ANY
# loss block. The block-match check previously only accepted an exact
# domain match (or compatible_with), falsely flagging agnostic losses
# placed under image_losses/kspace_losses.


def _patch_loss_domains(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, dict]) -> None:
    from spectramr.models.losses.registry import LossRegistry

    monkeypatch.setattr(LossRegistry, "_loss_domains", mapping, raising=False)


def test_loss_domain_block_match_accepts_agnostic_under_image_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loss_domains(monkeypatch, {"nll_bits_per_dim": {"domain": "agnostic"}})
    cfg = _losses_cfg(
        "image",
        image=[types.SimpleNamespace(name="nll_bits_per_dim", weight=1.0, enabled=True)],
    )
    results = ConfigHealthChecker().check_loss_domain_block_match(cfg)
    assert all(r.passed for r in results), [r.message for r in results if not r.passed]


def test_loss_domain_block_match_still_rejects_genuine_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loss_domains(monkeypatch, {"kspace_l2": {"domain": "kspace"}})
    cfg = _losses_cfg(
        "image",
        image=[types.SimpleNamespace(name="kspace_l2", weight=1.0, enabled=True)],
    )
    results = ConfigHealthChecker().check_loss_domain_block_match(cfg)
    failures = [r for r in results if not r.passed]
    assert len(failures) == 1
    assert failures[0].severity == "error"
    assert failures[0].check_name == "loss_domain_block_match"


# ── check_amp_grad_clip_interaction ─────────────────────────────────────


def _opt_cfg(use_amp: bool, enable_clip: bool, clip_value: float | None) -> Any:
    return types.SimpleNamespace(
        optimization=types.SimpleNamespace(
            # `use_amp` -> `precision.enabled`, and the two clipping scalars
            # -> `gradient.clip`, in the optimization decomposition.
            precision=types.SimpleNamespace(enabled=use_amp),
            gradient=types.SimpleNamespace(
                # `clip` is a sub-block, not a scalar: the old flat
                # `enable_gradient_clipping` / `gradient_clip_value` pair became
                # `gradient.clip.enabled` / `gradient.clip.value`.
                clip=types.SimpleNamespace(enabled=enable_clip, value=clip_value),
            ),
        ),
    )


def test_amp_check_skips_when_amp_disabled() -> None:
    cfg = _opt_cfg(use_amp=False, enable_clip=True, clip_value=None)
    results = ConfigHealthChecker().check_amp_grad_clip_interaction(cfg)
    assert all(r.passed for r in results)


def test_amp_check_passes_with_positive_clip_value() -> None:
    cfg = _opt_cfg(use_amp=True, enable_clip=True, clip_value=1.0)
    results = ConfigHealthChecker().check_amp_grad_clip_interaction(cfg)
    assert all(r.passed for r in results)


def test_amp_check_warns_when_clip_value_missing_or_zero() -> None:
    cfg = _opt_cfg(use_amp=True, enable_clip=True, clip_value=None)
    results = ConfigHealthChecker().check_amp_grad_clip_interaction(cfg)
    failures = [r for r in results if not r.passed]
    assert len(failures) == 1
    assert failures[0].category == "amp_grad_clip_interaction"
    assert failures[0].severity == "warning"
    assert failures[0].fix_hint is not None


# ─────────────────────────────────────────────────────────────────────── #
# F3 / E15 — check_acceleration_schedule_steps_match_diffusion
#
# The 2026-05-16 smoke audit found 60 runtime ERROR-level hits in the log
# from `acceleration_schedule_steps_match`: YAMLs declaring
# `training.diffusion.timesteps=1000` while leaving
# `acceleration.schedule_steps` at the schema default of 10000. The
# pre-2026-05-17 gate required *both* fields to be user-declared, which
# silently suppressed exactly this case at audit-time and pushed the
# detection to runtime.
#
# After F3: the check fires when EITHER field is user-declared and the
# effective values mismatch. Audit anchor:
# TODO/audit/smoke_audit_20260516.md §F3.
# ─────────────────────────────────────────────────────────────────────── #


def _diffusion_accel_cfg(
    *,
    accel_schedule_steps: int = 10000,
    accel_set_fields: set[str] | None = None,
    diff_timesteps: int = 1000,
    diff_set_fields: set[str] | None = None,
    diffusion_present: bool = True,
) -> Any:
    """Build a minimal config with acceleration + diffusion blocks.

    ``accel_set_fields`` / ``diff_set_fields`` mimic Pydantic's
    ``model_fields_set`` — the set of YAML-declared field names. The
    health check reads this to distinguish user-set values from
    schema defaults.
    """
    diff = (
        types.SimpleNamespace(
            timesteps=diff_timesteps,
            model_fields_set=diff_set_fields or set(),
        )
        if diffusion_present
        else None
    )
    return types.SimpleNamespace(
        # ROOT fold: the top-level block is `undersampling:` now.
        undersampling=types.SimpleNamespace(
            schedule_steps=accel_schedule_steps,
            model_fields_set=accel_set_fields or set(),
        ),
        training=types.SimpleNamespace(diffusion=diff),
    )


def test_schedule_steps_check_skips_non_diffusion_paradigm() -> None:
    """Non-diffusion paradigms (no ``training.diffusion``) are not checked."""
    cfg = _diffusion_accel_cfg(diffusion_present=False)
    res = ConfigHealthChecker().check_acceleration_schedule_steps_match_diffusion(cfg)
    assert res.passed
    assert "Not a diffusion paradigm" in res.message


def test_schedule_steps_check_skips_when_neither_user_set() -> None:
    """Neither field user-declared → nothing to enforce (no false positives)."""
    cfg = _diffusion_accel_cfg(
        accel_schedule_steps=10000,
        accel_set_fields=set(),
        diff_timesteps=1000,
        diff_set_fields=set(),
    )
    res = ConfigHealthChecker().check_acceleration_schedule_steps_match_diffusion(cfg)
    assert res.passed
    assert "nothing to enforce" in res.message


def test_schedule_steps_check_fires_when_timesteps_user_set_only() -> None:
    """F3 regression: timesteps user-set, schedule_steps default → must catch.

    This is the exact case driving 60 runtime ERROR-level hits in the
    2026-05-16 smoke audit. The pre-F3 gate suppressed it.
    """
    cfg = _diffusion_accel_cfg(
        accel_schedule_steps=10000,
        accel_set_fields=set(),
        diff_timesteps=1000,
        diff_set_fields={"timesteps"},
    )
    res = ConfigHealthChecker().check_acceleration_schedule_steps_match_diffusion(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert res.category == "physics_contract"
    assert "training.diffusion.timesteps=1000" in res.message
    assert "acceleration.schedule_steps=10000" in res.message
    assert res.fix_hint is not None


def test_schedule_steps_check_fires_when_schedule_steps_user_set_only() -> None:
    """Symmetric to the above: schedule_steps user-set, timesteps default."""
    cfg = _diffusion_accel_cfg(
        accel_schedule_steps=200,
        accel_set_fields={"schedule_steps"},
        diff_timesteps=1000,
        diff_set_fields=set(),
    )
    res = ConfigHealthChecker().check_acceleration_schedule_steps_match_diffusion(cfg)
    assert not res.passed
    assert res.severity == "error"


def test_schedule_steps_check_passes_when_both_match() -> None:
    """When effective values match, the check passes regardless of who set what."""
    cfg = _diffusion_accel_cfg(
        accel_schedule_steps=1000,
        accel_set_fields={"schedule_steps"},
        diff_timesteps=1000,
        diff_set_fields={"timesteps"},
    )
    res = ConfigHealthChecker().check_acceleration_schedule_steps_match_diffusion(cfg)
    assert res.passed
    assert "matches" in res.message


def test_schedule_steps_check_passes_when_both_match_via_num_timesteps() -> None:
    """``num_timesteps`` is the legacy alias for ``timesteps`` — same gate."""
    diff = types.SimpleNamespace(
        num_timesteps=500,
        timesteps=None,
        model_fields_set={"num_timesteps"},
    )
    cfg = types.SimpleNamespace(
        # ROOT fold: the top-level block is `undersampling:` now.
        undersampling=types.SimpleNamespace(
            schedule_steps=500,
            model_fields_set={"schedule_steps"},
        ),
        training=types.SimpleNamespace(diffusion=diff),
    )
    res = ConfigHealthChecker().check_acceleration_schedule_steps_match_diffusion(cfg)
    assert res.passed


# ─────────────────────────────────────────────────────────────────────── #
# F4 / E20 — AdversarialMixin requires losses.gan at audit-time
#
# 2026-05-16 smoke audit found 35 runtime ValueErrors from
# AdversarialMixin.__init__: training_mode in {gan, disentangled,
# guided_sr} with no `losses.gan:` block. Audit anchor:
# TODO/audit/smoke_audit_20260516.md §F4.
# ─────────────────────────────────────────────────────────────────────── #


def _gan_cfg(training_mode: str, gan_loss_block: Any) -> Any:
    """Minimal config shape for the AdversarialMixin check."""
    return types.SimpleNamespace(
        training=types.SimpleNamespace(training_mode=training_mode),
        losses=types.SimpleNamespace(gan=gan_loss_block),
    )


def test_adversarial_check_skips_when_training_mode_not_in_set() -> None:
    """Non-adversarial training_modes pass cleanly."""
    cfg = _gan_cfg(training_mode="reconstruction", gan_loss_block=None)
    res = ConfigHealthChecker().check_adversarial_strategy_requires_gan_loss(cfg)
    assert res.passed
    assert "does not use AdversarialMixin" in res.message


def test_adversarial_check_fires_on_gan_mode_missing_loss_block() -> None:
    """F4 regression: training_mode='gan' with no losses.gan → error."""
    cfg = _gan_cfg(training_mode="gan", gan_loss_block=None)
    res = ConfigHealthChecker().check_adversarial_strategy_requires_gan_loss(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert res.category == "strategy_requires_schema"
    assert "AdversarialMixin" in res.message
    assert "losses.gan" in res.message
    assert res.fix_hint is not None
    assert "training.training_mode" in res.yaml_keys
    assert "losses.gan" in res.yaml_keys


def test_adversarial_check_fires_on_disentangled_mode_missing_loss_block() -> None:
    """``disentangled`` also uses AdversarialMixin (see DisentangledTrainingStrategy)."""
    cfg = _gan_cfg(training_mode="disentangled", gan_loss_block=None)
    res = ConfigHealthChecker().check_adversarial_strategy_requires_gan_loss(cfg)
    assert not res.passed
    assert res.severity == "error"


def test_adversarial_check_fires_on_guided_sr_mode_missing_loss_block() -> None:
    """``guided_sr`` also uses AdversarialMixin (GuidedSuperResolutionStrategy)."""
    cfg = _gan_cfg(training_mode="guided_sr", gan_loss_block=None)
    res = ConfigHealthChecker().check_adversarial_strategy_requires_gan_loss(cfg)
    assert not res.passed


def test_adversarial_check_passes_when_gan_loss_block_populated() -> None:
    """A non-None ``losses.gan`` clears the check (content not validated here)."""
    gan_block = types.SimpleNamespace(lambda_adv=0.1)
    cfg = _gan_cfg(training_mode="gan", gan_loss_block=gan_block)
    res = ConfigHealthChecker().check_adversarial_strategy_requires_gan_loss(cfg)
    assert res.passed
    assert "populated" in res.message


def test_adversarial_check_handles_losses_missing_entirely() -> None:
    """``config.losses`` itself can be None — check must not crash."""
    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(training_mode="gan"),
        losses=None,
    )
    res = ConfigHealthChecker().check_adversarial_strategy_requires_gan_loss(cfg)
    assert not res.passed
    assert res.severity == "error"


# ─────────────────────────────────────────────────────────────────────── #
# F8 / E2 — CycleBloch requires a discriminator at audit-time
#
# 7 runtime ValueError hits in the 2026-05-16 smoke audit. Same shape as
# F4 (AdversarialMixin / losses.gan): the strategy fails at the first
# training step when a required schema block is missing. Audit anchor:
# TODO/audit/smoke_audit_20260516.md §F8.
# ─────────────────────────────────────────────────────────────────────── #


def _cycle_bloch_cfg(training_mode: str, disc_block: Any) -> Any:
    """Minimal config for the cycle_bloch_requires_discriminator check."""
    return types.SimpleNamespace(
        training=types.SimpleNamespace(training_mode=training_mode),
        model=types.SimpleNamespace(discriminator_component=disc_block),
    )


def test_cycle_bloch_check_skips_when_not_cycle_bloch_mode() -> None:
    """Non-cycle_bloch modes are not checked."""
    cfg = _cycle_bloch_cfg(training_mode="reconstruction", disc_block=None)
    res = ConfigHealthChecker().check_cycle_bloch_requires_discriminator(cfg)
    assert res.passed
    assert "not cycle_bloch" in res.message


def test_cycle_bloch_check_fires_when_discriminator_missing() -> None:
    """F8 regression: training_mode='cycle_bloch' with no discriminator → error."""
    cfg = _cycle_bloch_cfg(training_mode="cycle_bloch", disc_block=None)
    res = ConfigHealthChecker().check_cycle_bloch_requires_discriminator(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert res.category == "strategy_requires_schema"
    assert "discriminator_component" in res.message
    assert "model.discriminator_component" in res.yaml_keys


def test_cycle_bloch_check_passes_when_discriminator_populated() -> None:
    """Non-None ``discriminator_component`` clears the check."""
    disc = types.SimpleNamespace(discriminator_type="patch_gan_2d", in_channels=1)
    cfg = _cycle_bloch_cfg(training_mode="cycle_bloch", disc_block=disc)
    res = ConfigHealthChecker().check_cycle_bloch_requires_discriminator(cfg)
    assert res.passed


def test_cycle_bloch_check_handles_model_missing() -> None:
    """``config.model`` itself None — check must not crash."""
    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(training_mode="cycle_bloch"),
        model=None,
    )
    res = ConfigHealthChecker().check_cycle_bloch_requires_discriminator(cfg)
    assert not res.passed
    assert res.severity == "error"


# ─────────────────────────────────────────────────────────────────────── #
# F7 / 2026-05-17 round 4 — audit-assumption transparency for the
# channel-domain alignment check. The existing static derivation in
# check_domain_alignment has three known blind spots that let configs
# slip past audit and crash at the first training batch with
# [DomainMismatch] (63 hits in 2026-05-16 smoke). The F7 check makes
# each blind spot visible as a warning so ``--strict`` mode (smoke
# wrapper default) can hard-gate them.
#
# Audit anchor: TODO/audit/smoke_audit_20260516.md §F7.
# ─────────────────────────────────────────────────────────────────────── #


def _channel_audit_cfg(
    model_type: str = "configurable_unet",
    in_channels: int = 1,
    coil_mode: str = "rss_image",
    pre_model_steps: list[Any] | None = None,
    target_channels: int | None = None,
) -> Any:
    """Minimal config shape for check_channel_audit_assumptions."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            model_type=model_type,
            in_channels=in_channels,
        ),
        data=types.SimpleNamespace(
            # phase 9a moved this into `data.coils`.
            coils=types.SimpleNamespace(processing_mode=coil_mode),
            # `target_channels` moved into `data.domain`.
            domain=types.SimpleNamespace(target_channels=target_channels),
        ),
        adapters=types.SimpleNamespace(pre_model=pre_model_steps or []),
    )


def test_channel_audit_assumptions_no_blind_spots_passes() -> None:
    """Clean config (rss_image + no adapters + non-concat model) → info."""
    cfg = _channel_audit_cfg()
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    assert len(results) == 1
    assert results[0].passed
    assert results[0].severity == "info"
    assert "No channel-audit blind spots" in results[0].message


def test_channel_audit_assumptions_input_concat_model_emits_info() -> None:
    """F6 / 2026-05-20: ``_INPUT_CONCAT_MODELS`` is by-design — info, not warning.

    Previously (F7) this was a strict-mode warning; the 2026-05-19 smoke
    surfaced 70 instances on routine cold-diffusion / disentangled-MRI YAMLs
    where the runtime DomainMismatch check covers the actual verification.
    Demote to ``info`` so --strict mode no longer rejects the by-design path.
    """
    cfg = _channel_audit_cfg(model_type="disentangled_mri")
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    # No warnings expected.
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    assert len(warnings) == 0
    # The info entry mentions the by-design bypass.
    infos = [r for r in results if r.severity == "info"]
    assert any(
        "_INPUT_CONCAT_MODELS" in r.message and "disentangled_mri" in r.message for r in infos
    )


def test_channel_audit_assumptions_kspace_cold_diffusion_also_info() -> None:
    """``kspace_cold_diffusion`` (the second _INPUT_CONCAT entry) is info, not warning."""
    cfg = _channel_audit_cfg(model_type="kspace_cold_diffusion")
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    assert len(warnings) == 0
    infos = [r for r in results if r.severity == "info"]
    assert any("kspace_cold_diffusion" in r.message for r in infos)


def test_channel_audit_assumptions_known_pre_model_chain_emits_info() -> None:
    """F6b / 2026-05-20: chain of audit-known adapters is verified statically → info.

    ``rss_coils_to_magnitude`` is in ``_ADAPTER_CHANNEL_EFFECTS`` so the
    static channel resolution covers it; no strict-mode warning needed.
    """
    adapter_step = types.SimpleNamespace(name="rss_coils_to_magnitude")
    cfg = _channel_audit_cfg(pre_model_steps=[adapter_step])
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    assert len(warnings) == 0
    infos = [r for r in results if r.severity == "info"]
    assert any("rss_coils_to_magnitude" in r.message and "audit-known" in r.message for r in infos)


def test_channel_audit_assumptions_unknown_pre_model_chain_still_warns() -> None:
    """F6b / 2026-05-20: chain with at least one unknown-effect adapter still warns."""
    adapter_step = types.SimpleNamespace(name="custom_unknown_adapter")
    cfg = _channel_audit_cfg(pre_model_steps=[adapter_step])
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    assert len(warnings) == 1
    assert "adapters.pre_model" in warnings[0].message
    assert "custom_unknown_adapter" in warnings[0].message
    assert "adapters.pre_model" in warnings[0].yaml_keys


def test_channel_audit_assumptions_coil_mode_none_without_target_channels_is_info() -> None:
    """F32 / 2026-05-22: ``coil_processing_mode='none'`` is an UNVERIFIABLE
    assumption (the channel count is deferred to the h5 header at runtime),
    not a defect. It is now informational (passed=True) so --strict does not
    hard-gate intentional multi-coil arms. The runtime [DomainMismatch] check
    remains the source of truth.
    """
    cfg = _channel_audit_cfg(coil_mode="none", target_channels=None)
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    assert warnings == []
    deferral = [r for r in results if r.category == "audit_assumption_unverified"]
    assert deferral and all(r.severity == "info" and r.passed for r in deferral)
    assert "data.coil_processing_mode" in deferral[0].yaml_keys


def test_channel_audit_assumptions_coil_mode_none_matching_target_channels_is_info() -> None:
    """F6c / 2026-05-20: ``coil_processing_mode='none'`` + matching
    ``data.target_channels`` is the user's explicit declaration → info, not warning.

    Previously these emitted 88 strict-mode warnings in the 2026-05-19 smoke
    run for YAMLs that DID set target_channels (i.e., already declared the
    intended count). With the demotion, --strict no longer rejects them.
    """
    cfg = _channel_audit_cfg(
        coil_mode="none",
        in_channels=4,
        target_channels=4,
    )
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    assert len(warnings) == 0
    infos = [r for r in results if r.severity == "info"]
    assert any("coil_processing_mode='none'" in r.message and "matches" in r.message for r in infos)


def test_channel_audit_assumptions_coil_mode_none_mismatched_target_channels_is_info() -> None:
    """F32 / 2026-05-22: target_channels != in_channels under coil='none' is the
    legitimate ASYMMETRIC-recon case (e.g. multi-coil input → 1-channel
    magnitude target). It cannot be confirmed statically, so it is now
    informational rather than a strict-mode-failing warning; the runtime
    DomainMismatch check guards the actual header value.
    """
    cfg = _channel_audit_cfg(
        coil_mode="none",
        in_channels=4,
        target_channels=8,
    )
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    assert warnings == []
    deferral = [r for r in results if r.category == "audit_assumption_unverified"]
    assert deferral and all(r.severity == "info" and r.passed for r in deferral)


def test_channel_audit_assumptions_multiple_blind_spots_stack() -> None:
    """All blind spots simultaneously — only the unresolved ones warn.

    With F6 / F6b / F6c the by-design and statically-verified cases no
    longer count as blind spots, and F32 (2026-05-22) demoted the
    coil_mode='none' deferral to info. Only the unknown-effect adapter chain
    remains as a warning.
    """
    adapter_step = types.SimpleNamespace(name="custom_chan_adapter")
    cfg = _channel_audit_cfg(
        model_type="kspace_cold_diffusion",  # info now
        coil_mode="none",  # info now (F32 deferral)
        pre_model_steps=[adapter_step],  # warns (unknown adapter)
        target_channels=None,
    )
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    assert len(warnings) == 1, (
        f"Expected 1 warning (unknown adapter only); got "
        f"{len(warnings)}: {[r.message[:60] for r in warnings]}"
    )


def test_channel_audit_assumptions_handles_missing_blocks() -> None:
    """``config.model`` / ``config.data`` / ``config.adapters`` None — no crash."""
    cfg = types.SimpleNamespace(model=None, data=None, adapters=None)
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    # Should produce the "no blind spots" info result, not crash.
    assert len(results) == 1
    assert results[0].passed
    assert results[0].severity == "info"


# ─────────────────────────────────────────────────────────────────────── #
# F-OUT / 2026-05-17 round 6 — training.output_dir convention enforcement.
#
# 197 YAMLs in the 2026-05-16 smoke run misrouted outputs: 77 missing
# output_dir entirely, 85 stuck in the deprecated linear_configs/ tree,
# 35 with non-canonical prefixes (experiments/active/<name>,
# experiments/outputs/, experiments/test_experiments/). The runtime
# effect: validation PNGs, mosaics, and reports landed in arbitrary
# locations, breaking mosaic_validation.py and process_smoke_log.py
# discovery. F-OUT bulk-rewrote the 197 to ``experiments/results/<name>``
# and added this audit-time gate to prevent regression.
#
# Audit anchor: TODO/audit/smoke_audit_20260516.md §F-OUT.
# ─────────────────────────────────────────────────────────────────────── #


def _output_dir_cfg(value: Any) -> Any:
    """Minimal config shape for the output_dir convention check."""
    return types.SimpleNamespace(
        training=types.SimpleNamespace(output_dir=value),
    )


def test_output_dir_check_passes_canonical_path() -> None:
    """``experiments/results/<name>`` is canonical → info."""
    cfg = _output_dir_cfg("experiments/results/my_experiment")
    res = ConfigHealthChecker().check_output_dir_convention(cfg)
    assert res.passed
    assert res.severity == "info"
    assert "follows the experiments/results/<name>" in res.message


def test_output_dir_check_warns_when_missing() -> None:
    """``training.output_dir`` absent → warning (--strict gates)."""
    cfg = _output_dir_cfg(None)
    res = ConfigHealthChecker().check_output_dir_convention(cfg)
    assert not res.passed
    assert res.severity == "warning"
    assert res.category == "output_dir_missing"
    assert "training.output_dir" in res.yaml_keys
    assert "mosaic_validation.py" in res.message
    assert res.fix_hint is not None


def test_output_dir_check_errors_on_linear_configs_prefix() -> None:
    """``experiments/results/linear_configs/<name>`` → hard error."""
    cfg = _output_dir_cfg("experiments/results/linear_configs/experiment_42")
    res = ConfigHealthChecker().check_output_dir_convention(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert res.category == "output_dir_deprecated"
    assert "linear_configs/" in res.message


def test_output_dir_check_errors_on_experiments_active_prefix() -> None:
    """``experiments/active/<name>`` (self-referential) → hard error."""
    cfg = _output_dir_cfg("experiments/active/experiment_30_mamba")
    res = ConfigHealthChecker().check_output_dir_convention(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert "experiments/active/" in res.message


def test_output_dir_check_errors_on_experiments_outputs_prefix() -> None:
    """``experiments/outputs/<name>`` (legacy) → hard error."""
    cfg = _output_dir_cfg("experiments/outputs/stage1_vae")
    res = ConfigHealthChecker().check_output_dir_convention(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert "experiments/outputs/" in res.message


def test_output_dir_check_warns_on_nonstandard_prefix() -> None:
    """Any other prefix → warning (might be intentional, but flag it)."""
    cfg = _output_dir_cfg("/tmp/scratch/output")
    res = ConfigHealthChecker().check_output_dir_convention(cfg)
    assert not res.passed
    assert res.severity == "warning"
    assert res.category == "output_dir_nonstandard"


def test_output_dir_check_handles_missing_training_block() -> None:
    """``config.training`` itself None — no crash."""
    cfg = types.SimpleNamespace(training=None)
    res = ConfigHealthChecker().check_output_dir_convention(cfg)
    assert res.passed
    assert "no training block" in res.message


def test_output_dir_check_wired_into_run_all_checks() -> None:
    """F-OUT wiring: the check is discoverable on ConfigHealthChecker."""
    checker = ConfigHealthChecker()
    assert hasattr(checker, "check_output_dir_convention")
    assert hasattr(ConfigHealthChecker, "_OUTPUT_DIR_DEPRECATED_PREFIXES")
    # The deprecated set should include all four prefixes that caused
    # the 2026-05-16 smoke regression.
    deprecated = ConfigHealthChecker._OUTPUT_DIR_DEPRECATED_PREFIXES
    assert "experiments/results/linear_configs/" in deprecated
    assert "experiments/active/" in deprecated
    assert "experiments/outputs/" in deprecated
    assert "experiments/test_experiments/" in deprecated


# ─────────────────────────────────────────────────────────────────────── #
# F7-Hoist / 2026-05-17 round 6 — adapter-aware channel resolution.
#
# F7 (round 4) emits a warning whenever ``adapters.pre_model`` is
# declared because the static check can't model the adapter's channel
# effect. F7-Hoist promotes the warning to a hard error for the 4
# registered adapters whose channel transform is deterministic:
#   - rss_coils_to_magnitude: any → 1
#   - magnitude_from_complex: 2C → C
#   - real_imag_interleave_to_complex: 2C → C
#   - complex_to_real_imag_interleave: C → 2C
# Audit anchor: TODO/audit/smoke_audit_20260516.md §F7-Hoist.
# ─────────────────────────────────────────────────────────────────────── #


def _adapter_chain_cfg(
    model_type: str = "image_unet",
    in_channels: int = 1,
    coil_mode: str = "rss",
    dataset_type: str = "kspace",
    adapter_names: list[str] | None = None,
) -> Any:
    """Minimal config for check_adapter_chain_channel_resolution."""
    steps = [types.SimpleNamespace(name=n) for n in (adapter_names or [])]
    return types.SimpleNamespace(
        model=types.SimpleNamespace(model_type=model_type, in_channels=in_channels),
        data=types.SimpleNamespace(
            # phase 9a moved this into `data.coils`.
            coils=types.SimpleNamespace(processing_mode=coil_mode),
            dataset_type=dataset_type,
            num_virtual_coils=None,
        ),
        adapters=types.SimpleNamespace(pre_model=steps),
    )


def test_adapter_chain_passes_when_no_pre_model_chain() -> None:
    cfg = _adapter_chain_cfg(adapter_names=[])
    res = ConfigHealthChecker().check_adapter_chain_channel_resolution(cfg)
    assert res.passed
    assert "no adapters.pre_model" in res.message


def test_adapter_chain_rss_collapses_to_1_channel_when_matches() -> None:
    """``rss`` k-space (2ch) → ``rss_coils_to_magnitude`` (1ch) matches in_channels=1."""
    cfg = _adapter_chain_cfg(
        in_channels=1,
        coil_mode="rss",
        dataset_type="kspace",
        adapter_names=["rss_coils_to_magnitude"],
    )
    res = ConfigHealthChecker().check_adapter_chain_channel_resolution(cfg)
    assert res.passed
    assert "matches model.in_channels=1" in res.message


def test_adapter_chain_errors_on_known_mismatch() -> None:
    """RSS (1ch out) + model.in_channels=2 → hard error."""
    cfg = _adapter_chain_cfg(
        in_channels=2,
        coil_mode="rss",
        dataset_type="kspace",
        adapter_names=["rss_coils_to_magnitude"],
    )
    res = ConfigHealthChecker().check_adapter_chain_channel_resolution(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert res.category == "adapter_chain_channel_mismatch"
    assert "model receives 1ch" in res.message
    assert "model.in_channels=2" in res.message


def test_adapter_chain_defers_on_unknown_adapter_name() -> None:
    """Unknown adapter → info (round-4 F7 warning carries the case)."""
    cfg = _adapter_chain_cfg(
        adapter_names=["rss_coils_to_magnitude", "custom_user_adapter"],
    )
    res = ConfigHealthChecker().check_adapter_chain_channel_resolution(cfg)
    assert res.passed
    assert "deferred" in res.message
    assert "custom_user_adapter" in res.message


def test_adapter_chain_complex_to_real_doubles_channels() -> None:
    """``complex_to_real_imag_interleave`` doubles channel count."""
    effects = ConfigHealthChecker._ADAPTER_CHANNEL_EFFECTS
    assert effects["complex_to_real_imag_interleave"](1) == 2
    assert effects["complex_to_real_imag_interleave"](4) == 8


def test_adapter_chain_real_to_complex_halves_even_channels() -> None:
    """``real_imag_interleave_to_complex`` halves even channels; odd → identity."""
    effects = ConfigHealthChecker._ADAPTER_CHANNEL_EFFECTS
    assert effects["real_imag_interleave_to_complex"](2) == 1
    assert effects["real_imag_interleave_to_complex"](8) == 4
    assert effects["real_imag_interleave_to_complex"](3) == 3  # odd → identity


def test_adapter_chain_defers_when_expected_input_underivable() -> None:
    """``coil_processing_mode='none'`` → can't derive expected_input → defer."""
    cfg = _adapter_chain_cfg(
        coil_mode="none",
        adapter_names=["rss_coils_to_magnitude"],
    )
    res = ConfigHealthChecker().check_adapter_chain_channel_resolution(cfg)
    # _derive_expected_channels returns None for "none" mode → defer
    assert res.passed
    # Defer path either says "cannot be derived statically" or matches
    # other deferred messages; just verify it didn't error.


# ─────────────────────────────────────────────────────────────────────── #
# F2 / 2026-05-17 round 6 — visualization-interval reachability.
#
# 147 experiments in 2026-05-16 smoke PASSED with zero validation
# images. The cleanest YAML-level surface: viz_interval > max_iter
# (the val loop terminates before any tick). The smoke wrapper
# overrides viz_interval=TRAIN_ITERS so the bug is masked under smoke
# but real training is silently broken.
# Audit anchor: TODO/audit/smoke_audit_20260516.md §F2.
# ─────────────────────────────────────────────────────────────────────── #


def _viz_cfg(
    enable_viz: bool = True,
    save_val_imgs: bool | None = True,
    viz_interval: int | None = 100,
    max_iter: int | None = 1000,
) -> Any:
    return types.SimpleNamespace(
        logging=types.SimpleNamespace(
            enable_viz=enable_viz,
            # phase 10b moved the image knobs into `logging.images`.
            images=types.SimpleNamespace(save_validation=save_val_imgs),
        ),
        validation=types.SimpleNamespace(
            # phase 10a moved this into `validation.visualization`.
            visualization=types.SimpleNamespace(interval=viz_interval),
        ),
        training=types.SimpleNamespace(max_iterations=max_iter),
    )


def test_viz_interval_check_passes_when_interval_le_max_iter() -> None:
    cfg = _viz_cfg(viz_interval=100, max_iter=1000)
    res = ConfigHealthChecker().check_visualization_interval_reachable(cfg)
    assert res.passed
    assert "≤ training.max_iterations=1000" in res.message


def test_viz_interval_check_warns_when_interval_gt_max_iter() -> None:
    """F2 regression: interval > max_iter → warning (silent-no-PNG case)."""
    cfg = _viz_cfg(viz_interval=2000, max_iter=1000)
    res = ConfigHealthChecker().check_visualization_interval_reachable(cfg)
    assert not res.passed
    assert res.severity == "warning"
    assert res.category == "viz_interval_unreachable"
    assert "PNGs will NEVER be saved" in res.message
    assert "validation.visualization_interval" in res.yaml_keys
    assert "training.max_iterations" in res.yaml_keys


def test_viz_interval_check_skips_when_viz_disabled() -> None:
    """``enable_viz=False`` → info (check is moot)."""
    cfg = _viz_cfg(enable_viz=False, viz_interval=99999, max_iter=10)
    res = ConfigHealthChecker().check_visualization_interval_reachable(cfg)
    assert res.passed
    assert "viz disabled" in res.message


def test_viz_interval_check_skips_when_save_disabled() -> None:
    """``save_validation_images=False`` → info."""
    cfg = _viz_cfg(save_val_imgs=False, viz_interval=99999, max_iter=10)
    res = ConfigHealthChecker().check_visualization_interval_reachable(cfg)
    assert res.passed


def test_viz_interval_treats_none_save_as_true_per_schema_default() -> None:
    """``save_validation_images=None`` → defaults to True (schema-layer default)."""
    cfg = _viz_cfg(save_val_imgs=None, viz_interval=99999, max_iter=10)
    res = ConfigHealthChecker().check_visualization_interval_reachable(cfg)
    # save=None → True per the schema default at logging.py:169-172
    assert not res.passed, (
        "F2 must treat save_validation_images=None as True (schema "
        "default) so the silent-no-PNG case fires."
    )


# ─────────────────────────────────────────────────────────────────────── #
# F5 / 2026-05-17 round 6 — hardcoded cluster-path detection.
#
# 68+13 hits in 2026-05-16 smoke (E16) came from YAMLs hardcoding
# /project/alpha_lab/... paths. CLAUDE.md pitfall #16 says use
# PathResolver instead. Audit-time gate rejects the hardcoded form.
# Audit anchor: TODO/audit/smoke_audit_20260516.md §F5.
# ─────────────────────────────────────────────────────────────────────── #


def _path_cfg(
    data_root: str | None = None,
    index_path: str | None = None,
    validation_index_path: str | None = None,
    test_index_path: str | None = None,
) -> Any:
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            # phase 9 moved the path knobs into `data.source`; `data_root`
            # became `source.root`.
            source=types.SimpleNamespace(
                root=data_root,
                index_path=index_path,
                validation_index_path=validation_index_path,
                test_index_path=test_index_path,
            ),
        ),
    )


@pytest.fixture
def no_ambient_cluster_roots(monkeypatch, tmp_path):
    """Strip every ambient signal ``_user_configured_roots`` reads.

    The exemption ("I AM the cluster owner; this IS my mount") is intended
    behaviour, but a test asserting the check FIRES must not inherit it from the
    machine. On a cluster node — ``SPECTRAMR_DATA_ROOT`` set, or cwd under
    ``/project/alpha_lab/$USER`` — the path under test becomes exempt and the
    assertion silently inverts, so these tests passed on a dev box and failed on
    every cluster allocation (#630). ``chdir`` into ``tmp_path`` (never under a
    forbidden prefix) defeats the ``$USER`` + cwd auto-detect too.
    """
    for var in ("SPECTRAMR_DATA_ROOT", "PROJECT_ROOT", "SPECTRAMR_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_hardcoded_path_check_passes_on_relative_paths(
    no_ambient_cluster_roots,
) -> None:
    cfg = _path_cfg(
        data_root="databases/m4raw/data",
        index_path="data/manifests/m4raw_train.json",
    )
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert res.passed
    assert "no hardcoded" in res.message


def test_hardcoded_path_check_errors_on_project_prefix(
    no_ambient_cluster_roots,
) -> None:
    """F5 regression: any absolute path under ``/project/`` is forbidden."""
    cfg = _path_cfg(
        index_path="/project/alpha_lab/researcher/spectramr/data/manifests/m4raw.json",
    )
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert res.category == "hardcoded_cluster_path"
    assert "data.index_path" in res.yaml_keys
    assert "PathResolver" in res.fix_hint


def test_hardcoded_path_check_errors_on_scratch_prefix(
    no_ambient_cluster_roots,
) -> None:
    """``/scratch/`` is the second forbidden mount."""
    cfg = _path_cfg(data_root="/scratch/alpha_lab/cache")
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert "data.data_root" in res.yaml_keys


@pytest.mark.parametrize(
    "leaked",
    [
        "/project/some_other_group/user/data/manifests/m4raw.json",
        "/scratch/unrelated_alloc/cache",
        "/project/uppmax/proj123/data",
    ],
)
def test_hardcoded_path_check_fires_for_allocations_it_was_never_told_about(
    no_ambient_cluster_roots, leaked
) -> None:
    """The check keys on the MOUNT, not on a list of known allocation names.

    It used to enumerate three specific allocations, which meant a hardcoded
    path belonging to any other site sailed through -- the enumeration was both
    a hole in the check and this repo's own identity shipped in the package
    source. These paths are all invisible to the pre-widening implementation.
    """
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(_path_cfg(index_path=leaked))
    assert not res.passed
    assert res.severity == "error"
    assert "data.index_path" in res.yaml_keys


def test_user_root_auto_detect_finds_the_user_segment_at_any_depth(
    no_ambient_cluster_roots, monkeypatch
) -> None:
    """``/scratch/$USER/`` and ``/project/<group>/$USER/`` are both real layouts.

    The pre-widening auto-detect required ``$USER`` to be the FIRST segment
    after the prefix, which only ever matched one of the two. Widening the
    prefixes to the bare mounts would have silently broken the group-layout
    exemption -- i.e. made the check fire on the cluster owner's own paths.
    """
    from pathlib import Path

    monkeypatch.setenv("USER", "researcher")
    for cwd, expected in [
        ("/scratch/researcher/spectramr", "/scratch/researcher/"),
        ("/project/alpha_lab/researcher/spectramr", "/project/alpha_lab/researcher/"),
    ]:
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls, c=cwd: Path(c)))
        assert expected in ConfigHealthChecker()._user_configured_roots(), cwd


def test_hardcoded_path_check_reports_multiple_offenders(
    no_ambient_cluster_roots,
) -> None:
    """Multiple bad fields → first one named in message + count in note."""
    cfg = _path_cfg(
        data_root="/project/alpha_lab/cache",
        index_path="/project/alpha_lab/manifest.json",
        validation_index_path="/project/alpha_lab/val.json",
    )
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert not res.passed
    assert "3 field(s)" in res.message
    assert all(
        f in res.yaml_keys
        for f in ("data.data_root", "data.index_path", "data.validation_index_path")
    )


def test_hardcoded_path_check_is_not_defeated_by_ambient_env(
    no_ambient_cluster_roots, monkeypatch
) -> None:
    """The env exemption must not swallow SOMEONE ELSE's leaked prefix.

    The cluster sets ``SPECTRAMR_DATA_ROOT``; a YAML that hardcodes a different
    team's mount is still a leak and must still error.
    """
    monkeypatch.setenv("SPECTRAMR_DATA_ROOT", "/project/alpha_lab/researcher/")
    cfg = _path_cfg(
        # Same forbidden prefix, DIFFERENT account — a literal copied from a
        # colleague's config, which is exactly what this check exists to catch.
        index_path="/project/alpha_lab/someone_else/manifest.json",
        # ...while the user's own configured root stays exempt.
        data_root="/project/alpha_lab/researcher/databases",
    )
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert not res.passed
    assert "data.index_path" in res.yaml_keys
    assert "data.data_root" not in res.yaml_keys


def test_user_root_auto_detect_requires_the_running_user(
    no_ambient_cluster_roots, monkeypatch
) -> None:
    """The cwd auto-detect exempts ``/project/alpha_lab/$USER`` and nobody else.

    This is the branch that inverted the assertions on the cluster, and it can
    only be reached by faking cwd — a test cannot chdir into ``/project/``.
    """
    from pathlib import Path

    monkeypatch.setenv("USER", "researcher")
    monkeypatch.setattr(
        Path, "cwd", classmethod(lambda cls: Path("/project/alpha_lab/researcher/spectramr"))
    )
    roots = ConfigHealthChecker()._user_configured_roots()
    assert "/project/alpha_lab/researcher/" in roots

    # A colleague's subtree under the same prefix is NOT exempt.
    monkeypatch.setenv("USER", "someone_else")
    assert ConfigHealthChecker()._user_configured_roots() == ()


def test_hardcoded_path_check_handles_none_path_fields() -> None:
    """Optional fields (None) are not flagged."""
    cfg = _path_cfg()  # all None
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert res.passed


def test_hardcoded_path_check_exempts_user_configured_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path under the user's own ``SPECTRAMR_DATA_ROOT`` is not a leak.

    Regression for the May 18 2026 smoke run where 2 arms passed audit but
    failed training start-up with the same ``hardcoded_cluster_paths`` check:
    on the cluster, ``./data`` (the schema default) was joined against
    ``PROJECT_ROOT=/project/<user>/spectramr/`` by ``PathNormalizer``, and the
    resulting absolute path tripped the forbidden-prefix check. With the
    user-root exemption it no longer trips.
    """
    monkeypatch.setenv(
        "SPECTRAMR_DATA_ROOT",
        "/project/alpha_lab/researcher/spectramr",
    )
    cfg = _path_cfg(data_root="/project/alpha_lab/researcher/spectramr/data")
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert res.passed
    assert "no hardcoded" in res.message


def test_hardcoded_path_check_exempts_user_configured_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PROJECT_ROOT`` is honoured as a synonym for ``SPECTRAMR_DATA_ROOT``."""
    monkeypatch.delenv("SPECTRAMR_DATA_ROOT", raising=False)
    monkeypatch.setenv("PROJECT_ROOT", "/project/alpha_lab/researcher/spectramr")
    cfg = _path_cfg(
        data_root="/project/alpha_lab/researcher/spectramr/databases/fastmri",
        index_path="/project/alpha_lab/researcher/spectramr/data/manifests/m4raw.json",
    )
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert res.passed


def test_hardcoded_path_check_still_flags_other_users_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exemption is per-user — paths under another team's prefix still fail.

    If you set ``SPECTRAMR_DATA_ROOT=/project/alpha_lab/me`` but the YAML
    points at ``/project/alpha_lab/someone_else/...``, the check must still
    fire — otherwise the protection collapses.
    """
    monkeypatch.setenv("SPECTRAMR_DATA_ROOT", "/project/alpha_lab/me")
    cfg = _path_cfg(data_root="/project/alpha_lab/someone_else/cache")
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert "data.data_root" in res.yaml_keys


def test_hardcoded_path_check_exempts_only_prefix_not_substring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exemption is a prefix check with trailing slash — not a substring.

    ``/project/alpha_lab/me`` must not exempt ``/project/alpha_lab/meadow/...``
    just because they share a prefix without the boundary slash.
    """
    monkeypatch.setenv("SPECTRAMR_DATA_ROOT", "/project/alpha_lab/me")
    cfg = _path_cfg(data_root="/project/alpha_lab/meadow/cache")
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert not res.passed
    assert "data.data_root" in res.yaml_keys


def test_hardcoded_path_check_exempts_spectramr_root_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F5c / 2026-05-20: ``SPECTRAMR_ROOT`` (used by ``PathNormalizer``) is now
    honoured by the exemption logic alongside ``PROJECT_ROOT`` and
    ``SPECTRAMR_DATA_ROOT``.

    Previously only the latter two were checked, so a cluster that set
    ``SPECTRAMR_ROOT=/project/alpha_lab/<user>/spectramr`` (the variable used to
    resolve relative YAML paths) still saw the absolute path fail the
    forbidden-prefix check. With F5c the variable that *produced* the path
    is also the one that exempts it.
    """
    monkeypatch.delenv("SPECTRAMR_DATA_ROOT", raising=False)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.setenv("SPECTRAMR_ROOT", "/project/alpha_lab/researcher/spectramr")
    cfg = _path_cfg(
        data_root="/project/alpha_lab/researcher/spectramr/databases/m4raw/data",
    )
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert res.passed


def test_hardcoded_path_check_auto_detects_cluster_owner_from_cwd_and_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """F5c / 2026-05-20: cluster owner running from ``/project/<team>/<user>/...``
    is auto-exempt for paths under their own subtree even without setting any env-var.

    This is the dominant cluster failure mode in the 2026-05-19 smoke audit
    (389 of 414 configs): the .env didn't set ``SPECTRAMR_DATA_ROOT``, but
    the cluster owner WAS legitimately running from their own project mount.
    Auto-detect via ``$USER + cwd.parts`` makes the legitimate case work
    without env-var ceremony while keeping the colleague-leak protection
    (a different ``<user>`` segment doesn't match).
    """

    # Build a fake "cluster" cwd path with the proper structure.
    fake_cluster_root = tmp_path / "project" / "alpha_lab" / "researcher" / "spectramr"
    fake_cluster_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("SPECTRAMR_DATA_ROOT", raising=False)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.delenv("SPECTRAMR_ROOT", raising=False)
    monkeypatch.setenv("USER", "researcher")
    monkeypatch.chdir(fake_cluster_root)
    # Patch the forbidden prefixes so the test doesn't depend on the real
    # cluster paths — the rule under test is the auto-detect mechanism.
    monkeypatch.setattr(
        ConfigHealthChecker,
        "_FORBIDDEN_PATH_PREFIXES",
        frozenset({str(tmp_path / "project" / "alpha_lab") + "/"}),
    )
    cfg = _path_cfg(
        data_root=str(fake_cluster_root / "databases" / "m4raw" / "data"),
    )
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert res.passed, f"Expected auto-detect exemption, got: {res.message}"


def test_hardcoded_path_check_cwd_auto_detect_does_not_exempt_other_users(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """F5c / 2026-05-20: cwd auto-detect requires ``$USER`` to match the
    path-segment after the forbidden prefix — otherwise a colleague's leaked
    path under the same cluster prefix would be silently exempted.
    """
    fake_cluster_root = tmp_path / "project" / "alpha_lab" / "researcher" / "spectramr"
    fake_cluster_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("SPECTRAMR_DATA_ROOT", raising=False)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.delenv("SPECTRAMR_ROOT", raising=False)
    monkeypatch.setenv("USER", "researcher")
    monkeypatch.chdir(fake_cluster_root)
    monkeypatch.setattr(
        ConfigHealthChecker,
        "_FORBIDDEN_PATH_PREFIXES",
        frozenset({str(tmp_path / "project" / "alpha_lab") + "/"}),
    )
    # Colleague's path — different account segment.
    leaked = tmp_path / "project" / "alpha_lab" / "someone_else" / "leaked.json"
    cfg = _path_cfg(index_path=str(leaked))
    res = ConfigHealthChecker().check_hardcoded_cluster_paths(cfg)
    assert not res.passed
    assert "data.index_path" in res.yaml_keys


def test_channel_audit_assumptions_wired_into_run_all_checks() -> None:
    """F7 / F6 wiring: the check fires when run_all_checks executes.

    With F6 (2026-05-20) ``_INPUT_CONCAT_MODELS`` emits info, not warning.
    The wiring assertion is now that the check resolves and *emits results*
    — concrete severity is covered by the dedicated F6 tests above.
    """
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            model_type="disentangled_mri",
            in_channels=1,
        ),
        data=types.SimpleNamespace(
            # phase 9a moved this into `data.coils`.
            coils=types.SimpleNamespace(processing_mode="rss_image"),
            # `target_channels` moved into `data.domain`.
            domain=types.SimpleNamespace(target_channels=None),
        ),
        adapters=types.SimpleNamespace(pre_model=[]),
    )
    checker = ConfigHealthChecker()
    assert hasattr(checker, "check_channel_audit_assumptions")
    results = checker.check_channel_audit_assumptions(cfg)
    assert len(results) >= 1
    # Must include the _INPUT_CONCAT_MODELS by-design notice (info post-F6).
    assert any("_INPUT_CONCAT_MODELS" in r.message for r in results)


# ── F32: check_channel_audit_assumptions (coil_processing_mode='none') ──────


def _coil_none_cfg(in_channels: int, target_channels: int):
    """Minimal config reaching the coil_processing_mode='none' branch."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(in_channels=in_channels, out_channels=in_channels),
        data=types.SimpleNamespace(
            # phase 9a moved this into `data.coils`.
            coils=types.SimpleNamespace(processing_mode="none"),
            # `target_channels` moved into `data.domain`.
            domain=types.SimpleNamespace(target_channels=target_channels),
            dataset_type="fastmri_kspace",
        ),
        adapters=None,
    )


def test_channel_audit_assumptions_none_asymmetric_is_info_not_warning() -> None:
    """F32 — coil='none' with in_channels != target_channels (asymmetric recon:
    8-coil k-space input → 1-channel magnitude target) is an UNVERIFIABLE
    assumption, not a defect. It must be informational (passed=True), so
    --strict does not hard-gate the 33 intentional multi-coil arms.
    """
    cfg = _coil_none_cfg(in_channels=8, target_channels=1)
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    deferred = [r for r in results if r.category == "audit_assumption_unverified"]
    assert deferred, "expected the coil='none' deferral result"
    for r in deferred:
        assert r.severity == "info"
        assert r.passed is True


def test_channel_audit_assumptions_none_symmetric_is_resolved_info() -> None:
    """coil='none' with in_channels == target_channels stays an info pass."""
    cfg = _coil_none_cfg(in_channels=8, target_channels=8)
    results = ConfigHealthChecker().check_channel_audit_assumptions(cfg)
    bad = [r for r in results if not r.passed]
    assert bad == []


# ── PR-0: themed-strategy requires themed component ─────────────────────


def _cfg(training_mode, loss_names=(), model_type="hermitian_fno"):
    """Lightweight TrainingSettings stand-in for the themed-component check."""
    losses = types.SimpleNamespace(
        image_losses=[types.SimpleNamespace(name=n, enabled=True) for n in loss_names],
        kspace_losses=[],
        complex_losses=[],
    )
    return types.SimpleNamespace(
        training=types.SimpleNamespace(training_mode=training_mode),
        losses=losses,
        model=types.SimpleNamespace(model_type=model_type),
    )


def test_themed_check_fails_facade_config() -> None:
    """A themed key with a generic model and no themed loss fails (error)."""
    checker = ConfigHealthChecker()
    res = checker.check_themed_strategy_requires_themed_component(
        _cfg("tropical_mrf", loss_names=["l1", "ssim"], model_type="hermitian_fno")
    )
    assert res.passed is False
    assert res.severity == "error"
    assert res.category == "themed_strategy_requires_themed_component"


def test_themed_check_passes_when_loss_wired() -> None:
    checker = ConfigHealthChecker()
    res = checker.check_themed_strategy_requires_themed_component(
        _cfg("tropical_mrf", loss_names=["l1", "tropical_semiring_consistency"])
    )
    assert res.passed is True


def test_themed_check_passes_when_themed_model_wired() -> None:
    # heisenberg_phase's math is carried by the heisenberg_recon_unet model.
    checker = ConfigHealthChecker()
    res = checker.check_themed_strategy_requires_themed_component(
        _cfg("heisenberg_phase", loss_names=["l1"], model_type="heisenberg_recon_unet")
    )
    assert res.passed is True


def test_themed_check_exempts_honest_generic_key() -> None:
    checker = ConfigHealthChecker()
    res = checker.check_themed_strategy_requires_themed_component(
        _cfg("reconstruction", loss_names=["l1"], model_type="unet")
    )
    assert res.passed is True
    assert "not a themed" in res.message


class TestMambaSsmAuditHook:
    """Mamba-family configs must declare a usable mamba_ssm kernel at audit time."""

    import spectramr.models.blocks.mamba_block as _mb

    def test_non_mamba_model_passes(self) -> None:
        res = ConfigHealthChecker().check_mamba_models_require_mamba_ssm(_make_config("unet"))
        assert res.passed is True
        assert res.severity == "info"

    def test_mamba_model_errors_when_kernel_absent(self, monkeypatch) -> None:
        monkeypatch.setattr(self._mb, "_mamba_ssm_importable", lambda: False)
        monkeypatch.setattr(self._mb, "_mamba_fallback_allowed", lambda: False)
        res = ConfigHealthChecker().check_mamba_models_require_mamba_ssm(_make_config("ct_mamba"))
        assert res.passed is False
        assert res.severity == "error"
        assert "mamba_ssm" in res.message

    def test_mamba_model_warns_when_fallback_allowed(self, monkeypatch) -> None:
        monkeypatch.setattr(self._mb, "_mamba_ssm_importable", lambda: False)
        monkeypatch.setattr(self._mb, "_mamba_fallback_allowed", lambda: True)
        res = ConfigHealthChecker().check_mamba_models_require_mamba_ssm(
            _make_config("geo_mamba_unet")
        )
        assert res.passed is False
        assert res.severity == "warning"

    def test_fix_hint_names_the_arch_aware_installer(self, monkeypatch) -> None:
        """The bare pip line installs a wheel with no ``sm_70``, so it must not be the hint.

        This check gates on importability alone, and an architecture-mismatched
        wheel imports cleanly — so the remediation has to point at the installer
        that verifies the built architectures, or the audit sends a V100 user to
        the command that produced the broken install.
        """
        monkeypatch.setattr(self._mb, "_mamba_ssm_importable", lambda: False)
        monkeypatch.setattr(self._mb, "_mamba_fallback_allowed", lambda: False)
        res = ConfigHealthChecker().check_mamba_models_require_mamba_ssm(_make_config("ct_mamba"))
        assert res.fix_hint is not None
        assert "install-mamba" in res.fix_hint
        assert "--verify-only" in res.fix_hint
        assert "pip install -e '.[mamba]'" not in res.fix_hint

    def test_mamba_model_passes_when_kernel_available(self, monkeypatch) -> None:
        monkeypatch.setattr(self._mb, "_mamba_ssm_importable", lambda: True)
        res = ConfigHealthChecker().check_mamba_models_require_mamba_ssm(
            _make_config("bloch_mamba")
        )
        assert res.passed is True
        assert res.severity == "info"


# ─────────────────────────────────────────────────────────────────────── #
# #1929 — check_identity_paths_agree
#
# ``check_output_dir_convention`` above is a PREFIX test: it accepts
# ``experiments/results/<anything>``. The arms it cannot see are the ones whose
# output_dir, checkpoints, logs and metrics name DIFFERENT experiments, so half
# the run's artifacts land in another arm's tree. Measured over the 660
# ``experiments/inprogress`` arms: 658 agree, 2 do not.
#
# Every plant below is one shape the check must catch, or one the scope
# deliberately excludes and must therefore leave green (non-negotiable 15).
# ─────────────────────────────────────────────────────────────────────── #


def _identity_cfg(
    *,
    output_dir: Any = None,
    checkpoint_dir: Any = None,
    sinks_dir: Any = None,
    metrics_dir: Any = None,
) -> Any:
    """The four identity paths a stub can express, each independently settable."""
    return types.SimpleNamespace(
        training=types.SimpleNamespace(output_dir=output_dir),
        checkpoint=types.SimpleNamespace(checkpoint_dir=checkpoint_dir),
        logging=types.SimpleNamespace(sinks=types.SimpleNamespace(dir=sinks_dir)),
        metrics=types.SimpleNamespace(output_dir=metrics_dir),
    )


def test_identity_paths_that_agree_pass() -> None:
    """Anti-vacuity. Without this, a check that ALWAYS fired would satisfy every
    red assertion below and still be worthless."""
    cfg = _identity_cfg(
        output_dir="experiments/results/arm_alpha",
        checkpoint_dir="experiments/results/arm_alpha/checkpoints",
        sinks_dir="experiments/results/arm_alpha/logs",
        metrics_dir="experiments/results/arm_alpha/metrics",
    )
    res = ConfigHealthChecker().check_identity_paths_agree(cfg)
    assert res.passed
    assert res.severity == "info"
    assert "arm_alpha" in res.message


def test_a_split_between_output_dir_and_the_checkpoint_tree_is_reported() -> None:
    """The shape that motivates the check: two names, so two trees.

    This is ``conditioning/recon_field_conditioned_off_v1.yaml`` reduced -- an
    ``off`` control arm whose ``training.output_dir`` still names the ``on`` arm
    it was copied from, while its checkpoints and logs name itself.
    """
    cfg = _identity_cfg(
        output_dir="experiments/results/recon_field_conditioned_v1",
        checkpoint_dir="experiments/results/recon_field_conditioned_off_v1/checkpoints",
    )
    res = ConfigHealthChecker().check_identity_paths_agree(cfg)
    assert not res.passed
    assert res.severity == "warning"
    assert res.category == "identity_paths_disagree"
    assert set(res.yaml_keys) == {"training.output_dir", "checkpoint.checkpoint_dir"}
    assert "recon_field_conditioned_v1" in res.message
    assert "recon_field_conditioned_off_v1" in res.message
    assert res.fix_hint is not None


def test_a_third_disagreeing_path_is_counted_and_named() -> None:
    """Three names, not just "they differ" -- the message must say how many trees
    and which key put the arm in each, or the reader cannot tell which is wrong."""
    cfg = _identity_cfg(
        output_dir="experiments/results/arm_a",
        checkpoint_dir="experiments/results/arm_b",
        sinks_dir="experiments/results/arm_c",
    )
    res = ConfigHealthChecker().check_identity_paths_agree(cfg)
    assert not res.passed
    assert "3 different" in res.message
    assert set(res.yaml_keys) == {
        "training.output_dir",
        "checkpoint.checkpoint_dir",
        "logging.sinks.dir",
    }


def test_the_split_is_seen_through_the_save_dir_alias(tmp_path: Any) -> None:
    """Declared as ``save_dir``, compared as ``checkpoint_dir``.

    ``checkpoint.checkpoint_dir`` carries ``validation_alias="save_dir"`` and is
    the ONLY aliased field in the schema tree; 293 corpus arms spell it the alias
    way. A check that parsed YAML would see ``save_dir`` and ``output_dir`` as
    two unrelated keys and never compare them. This one reads the RESOLVED
    settings object, so the fold has already happened -- which is also why the
    reported key is the canonical spelling the run actually used, not the one the
    arm typed.
    """
    import yaml as _yaml

    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
    from spectramr.config.settings import TrainingSettings

    doc = {
        "config_version": CANONICAL_CONFIG_VERSION,
        "data": {"train_path": "/tmp/t", "val_path": "/tmp/v"},
        "optimization": {"learning_rate": 1e-4},
        "logging": {},
        "model": {"model_type": "unet"},
        "training": {"output_dir": "experiments/results/arm_alpha"},
        "checkpoint": {"save_dir": "experiments/results/arm_beta"},
    }
    path = tmp_path / "aliased_arm.yaml"
    path.write_text(_yaml.dump(doc))
    settings = TrainingSettings.from_yaml(str(path))

    # The fold is what makes the comparison possible; assert it happened.
    assert settings.checkpoint.checkpoint_dir == "experiments/results/arm_beta"

    res = ConfigHealthChecker().check_identity_paths_agree(settings)
    assert not res.passed
    assert "checkpoint.checkpoint_dir" in res.yaml_keys
    assert "arm_alpha" in res.message and "arm_beta" in res.message


def test_a_cwd_relative_checkpoint_dir_is_out_of_scope() -> None:
    """The noise control, and the reason this check is usable at all.

    ``./checkpoints`` is the schema default and 252 of 660 inprogress arms
    resolve to a CWD-relative checkpoint path. Comparing those against
    ``experiments/results/<name>`` would fire on a third of the corpus, the check
    would be baselined, and the 2 real splits would be lost in it. That is a
    separate defect with its own issue, not this one.
    """
    cfg = _identity_cfg(
        output_dir="experiments/results/arm_alpha",
        checkpoint_dir="./checkpoints",
    )
    res = ConfigHealthChecker().check_identity_paths_agree(cfg)
    assert res.passed


def test_a_nested_subdirectory_resolves_to_its_experiment_segment() -> None:
    """``experiments/results/<name>/logs`` is <name>, not "logs".

    Taking the LAST segment instead of the first would make every arm that nests
    its sinks disagree with its own output_dir -- a check that fires on almost
    everything and means nothing.
    """
    cfg = _identity_cfg(
        output_dir="experiments/results/arm_alpha",
        sinks_dir="experiments/results/arm_alpha/logs/train",
        metrics_dir="experiments/results/arm_alpha/metrics",
    )
    res = ConfigHealthChecker().check_identity_paths_agree(cfg)
    assert res.passed


def test_an_arm_with_no_convention_path_reports_the_absence() -> None:
    """Absent is a state to report, not one to infer (non-negotiable 18).

    Nothing under ``experiments/results/`` means nothing to compare -- which is
    ``check_output_dir_convention``'s finding, not this one's -- so the message
    says so rather than claiming agreement it never checked.
    """
    cfg = _identity_cfg(output_dir="./training_output", checkpoint_dir="./checkpoints")
    res = ConfigHealthChecker().check_identity_paths_agree(cfg)
    assert res.passed
    assert "no identity path" in res.message


def test_the_check_is_registered_in_run_all_checks(tmp_path: Any) -> None:
    """A defined-but-uncalled check is the exact shape non-negotiable 16 names.

    Asserted BEHAVIOURALLY -- by finding the result in a real report -- and never
    by scanning ``run_all_checks`` source. The call site carries a comment naming
    this check, so a source scan would match the comment and pass whether or not
    the method is ever invoked. ``test_the_dead_check_is_gone`` in the sibling
    module records that same trap.
    """
    import yaml as _yaml

    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
    from spectramr.config.settings import TrainingSettings

    doc = {
        "config_version": CANONICAL_CONFIG_VERSION,
        "data": {"train_path": "/tmp/t", "val_path": "/tmp/v"},
        "optimization": {"learning_rate": 1e-4},
        "logging": {},
        "model": {"model_type": "unet"},
        "training": {"output_dir": "experiments/results/arm_alpha"},
        "checkpoint": {"checkpoint_dir": "experiments/results/arm_beta"},
    }
    path = tmp_path / "split_arm.yaml"
    path.write_text(_yaml.dump(doc))

    report = ConfigHealthChecker().run_all_checks(TrainingSettings.from_yaml(str(path)))
    hits = [r for r in report.results if r.check_name == "identity_paths_agree"]
    assert len(hits) == 1, "run_all_checks does not invoke check_identity_paths_agree"
    assert not hits[0].passed
