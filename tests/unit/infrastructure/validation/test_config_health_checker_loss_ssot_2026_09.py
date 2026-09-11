"""Planted violations for ``check_loss_weight_declaration_ssot`` (non-negotiable 15).

A loss weight has two declaration surfaces: ``losses.<section>.lambda_<name>`` and
``losses.{image,kspace,complex,latent}_losses[].weight``. Since v6.0 the **domain
list is the SSOT** — only a list entry builds the loss module and selects the FFT
bridge, so a ``lambda_`` beside a list entry contributes nothing and is free to
drift away from the weight that trains.

Nothing detected that shape before this check: the duplicates load, resolve to a
single weight, and train correctly *today*, which is precisely why the redundancy
survives review and why editing the wrong surface later is silent.

Every fixture below is the real ``losses:`` block of a corpus arm, read from the
arm and frozen here rather than hand-written — a hand-written fixture that agrees
with a hand-written expectation tests nothing. ``LossConfigSchema(**block)`` was
verified to reproduce ``TrainingSettings.from_yaml``'s weight table entry-for-entry
on all three source arms (2026-09-07), which matters because
``build_loss_weight_table`` keys on ``model_fields_set``: a construction path that
pre-filled defaults would make these tests pass for the wrong reason.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fixtures, frozen from the producer
# ---------------------------------------------------------------------------

#: ``losses:`` of ``kspace_filling/attention_shootout/experiment_11_attention_none.yaml``
#: as it stood before this PR. Five names appear on both surfaces; ``hfen`` is
#: list-only, ``null_space_content`` is list-only and disabled, and
#: ``lambda_pre_dc_kspace`` / ``lambda_perceptual`` are lambda-only.
EXP11_LOSSES_BEFORE = {
    "reconstruction": {
        "lambda_complex_l1": 1.0,
        "lambda_log_spectral": 0.1,
        "lambda_sobolev_kspace": 0.05,
        "lambda_sense_adjoint_l1": 0.3,
        "log_spectral_skip_fft": True,
        "lambda_pre_dc_kspace": 0.3,
        "lambda_perceptual": 0.0,
    },
    "physics": {
        "lambda_complex_spatial_gradient": 1.0,
        "lambda_bloch_residual": 0.0,
        "lambda_physics_constraint": 0.0,
        "lambda_snr_preserving": 0.0,
    },
    "image_losses": [
        {
            "name": "hfen",
            "weight": 0.3,
            "enabled": True,
            "kwargs": {"kernel_size": 15, "sigma": 1.5, "normalize": True},
        }
    ],
    "kspace_losses": [
        {"name": "complex_l1", "weight": 1.0, "enabled": True},
        {"name": "log_spectral", "weight": 0.1, "enabled": True},
        {"name": "sobolev_kspace", "weight": 0.05, "enabled": True},
        {"name": "complex_spatial_gradient", "weight": 1.0, "enabled": True},
        {"name": "sense_adjoint_l1", "weight": 0.3, "enabled": True},
        {"name": "null_space_content", "weight": 0.25, "enabled": False},
    ],
    "complex_losses": [],
    "policy": {"output_domain": "kspace"},
}

#: The five names that carry a declaration on both surfaces, and the section each
#: redundant lambda lives in.
EXP11_DUPLICATES = {
    "complex_l1": "losses.reconstruction",
    "complex_spatial_gradient": "losses.physics",
    "log_spectral": "losses.reconstruction",
    "sense_adjoint_l1": "losses.reconstruction",
    "sobolev_kspace": "losses.reconstruction",
}

#: ``losses:`` of ``promoted/exp_promoted_mri_slam.yaml`` — the only corpus arm
#: declaring one canonical name in two *different lists*. That is a domain
#: decision (which FFT bridge the term uses), not the lambda/list redundancy this
#: check owns, so it must not be flagged here.
SLAM_LOSSES = {
    "image_losses": [{"name": "l1", "weight": 10.0, "enabled": True}],
    "kspace_losses": [{"name": "l1", "weight": 10.0, "enabled": True}],
    "complex_losses": [],
    "policy": {"output_domain": "kspace"},
}


def _settings(losses_block: dict):
    """A minimal ``TrainingSettings`` carrying only the losses block under test."""
    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.config.schemas.logging import LoggingConfigSchema
    from spectramr.config.schemas.loss import LossConfigSchema
    from spectramr.config.schemas.metrics import MetricsConfigSchema
    from spectramr.config.schemas.model import ModelConfigSchema
    from spectramr.config.schemas.optimization import OptimizationConfigSchema
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings(
        model=ModelConfigSchema(),
        data=DataConfigSchema(),
        optimization=OptimizationConfigSchema(),
        logging=LoggingConfigSchema(),
        metrics=MetricsConfigSchema(),
        losses=LossConfigSchema(**losses_block),
    )


def _failures(losses_block: dict):
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()
    return [
        r
        for r in checker.check_loss_weight_declaration_ssot(_settings(losses_block))
        if not r.passed
    ]


# ---------------------------------------------------------------------------
# The violation the check exists for
# ---------------------------------------------------------------------------


class TestDualSurfaceDeclarationIsFlagged:
    def test_exp11_before_names_exactly_the_five_duplicates(self):
        """The producing arm: 5 names, no more, no fewer."""
        failures = _failures(EXP11_LOSSES_BEFORE)
        assert len(failures) == 1, f"expected one result, got {failures}"
        message = failures[0].message
        for name, section in EXP11_DUPLICATES.items():
            assert f"'{name}'" in message, f"{name} not named in:\n{message}"
            assert section in message, f"{section} not named for {name}"
        # ``hfen`` and ``null_space_content`` are list-only; ``pre_dc_kspace`` and
        # ``perceptual`` are lambda-only. None is a duplicate.
        for single_surface in ("hfen", "null_space_content", "pre_dc_kspace"):
            assert f"'{single_surface}'" not in message, (
                f"'{single_surface}' has one declaration surface and must not be "
                f"reported as duplicated:\n{message}"
            )
        assert message.lstrip().startswith("5 loss weight(s)"), message

    def test_result_is_a_duplication_warning_not_an_error(self):
        """The arms train correctly today; the defect is the drift they permit."""
        result = _failures(EXP11_LOSSES_BEFORE)[0]
        assert result.check_name == "loss_weight_declaration_ssot"
        assert result.severity == "warning"
        assert result.category == "duplication"

    def test_fix_hint_names_the_winning_surface_and_forbids_the_reverse(self):
        """Deleting the list entry instead would stop the loss being built at all."""
        hint = _failures(EXP11_LOSSES_BEFORE)[0].fix_hint or ""
        assert "Delete the lambda_" in hint
        assert "Do NOT do the reverse" in hint

    @pytest.mark.parametrize(
        "lambda_site,list_name",
        [
            # Aliases collapse onto one canonical name, so the duplication is real
            # even when the two surfaces are spelled differently.
            ({"reconstruction": {"lambda_l2": 1.0}}, "mse"),
            # ``{"diffusion": {"lambda_mse": ...}}`` also aliases onto ``l2`` but is
            # NOT a duplicate -- a computer reads it as the diffusion-term weight.
            # Its behaviour is pinned in TestComputerResolvedLambdaIsExempt.
        ],
    )
    def test_alias_spellings_are_recognised_as_one_loss(self, lambda_site, list_name):
        block = {
            **lambda_site,
            "image_losses": [{"name": list_name, "weight": 1.0, "enabled": True}],
            "policy": {"output_domain": "image"},
        }
        failures = _failures(block)
        assert len(failures) == 1, f"alias duplication missed: {block}"
        assert "'l2'" in failures[0].message

    def test_disabled_list_entry_beside_a_live_lambda_is_still_flagged(self):
        """Deliberately not filtered on ``spec.enabled``.

        A disabled entry is the one an author re-enables later, and it is then the
        list weight that trains — not the lambda sitting beside it.
        """
        block = {
            "reconstruction": {"lambda_ssim": 0.1},
            "image_losses": [{"name": "ssim", "weight": 0.1, "enabled": False}],
            "policy": {"output_domain": "image"},
        }
        failures = _failures(block)
        assert len(failures) == 1
        assert "'ssim'" in failures[0].message

    def test_numerically_conflicting_surfaces_report_rather_than_abort_the_audit(self):
        """``build_loss_weight_table`` raises here; the check must not propagate it.

        An escaping ``ConfigurationError`` would take down every later check in the
        run, turning one bad arm into a blank report (#9: no silent gaps).
        """
        block = {
            "reconstruction": {"lambda_ssim": 0.5},
            "image_losses": [{"name": "ssim", "weight": 0.1, "enabled": False}],
            "policy": {"output_domain": "image"},
        }
        failures = _failures(block)
        assert len(failures) == 1
        assert failures[0].severity == "error"
        assert "could not be built" in failures[0].message


# ---------------------------------------------------------------------------
# The shapes the check must stay quiet on
# ---------------------------------------------------------------------------


class TestSingleSurfaceDeclarationsAreNotFlagged:
    def test_list_only_declaration_is_clean(self):
        block = {
            "kspace_losses": [{"name": "null_space_content", "weight": 0.25, "enabled": False}],
            "policy": {"output_domain": "kspace"},
        }
        assert _failures(block) == []

    def test_same_name_in_two_different_lists_is_a_domain_choice_not_a_duplicate(self):
        """``promoted/exp_promoted_mri_slam.yaml``.

        Both declarations are on the winning surface. Which list a term sits in
        selects its FFT bridge, so this is a deliberate two-domain objective.
        """
        assert _failures(SLAM_LOSSES) == []

    def test_migrated_exp11_is_clean(self):
        """The same arm with the five redundant lambdas deleted — the target state."""
        block = {
            **EXP11_LOSSES_BEFORE,
            "reconstruction": {
                "log_spectral_skip_fft": True,
                "lambda_pre_dc_kspace": 0.3,
                "lambda_perceptual": 0.0,
            },
            "physics": {
                "lambda_bloch_residual": 0.0,
                "lambda_physics_constraint": 0.0,
                "lambda_snr_preserving": 0.0,
            },
        }
        assert _failures(block) == []

    def test_deleting_the_lambdas_does_not_change_what_trains(self):
        """The migration must be weight-neutral, or it is a silent objective change."""
        from spectramr.models.losses.weights import build_loss_weight_table

        before = _settings(EXP11_LOSSES_BEFORE).losses
        after = _settings(
            {
                **EXP11_LOSSES_BEFORE,
                "reconstruction": {
                    "log_spectral_skip_fft": True,
                    "lambda_pre_dc_kspace": 0.3,
                    "lambda_perceptual": 0.0,
                },
                "physics": {
                    "lambda_bloch_residual": 0.0,
                    "lambda_physics_constraint": 0.0,
                    "lambda_snr_preserving": 0.0,
                },
            }
        ).losses

        def weights(losses):
            return {n: (s.weight, s.enabled) for n, s in build_loss_weight_table(losses).items()}

        assert weights(before) == weights(after)


class TestCheckIsWiredIntoTheAuditRun:
    def test_run_all_checks_dispatches_the_new_check(self):
        """A check nobody calls is a facade (#16)."""
        import inspect

        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        src = inspect.getsource(ConfigHealthChecker.run_all_checks)
        assert "check_loss_weight_declaration_ssot" in src


# ---------------------------------------------------------------------------
# A lambda a computer resolves is a knob, not a duplicate
# ---------------------------------------------------------------------------


#: Both cohort arms with this shape (``experiment_130_ti_ccd``,
#: ``experiment_cross_contrast_kspace_diffusion``) declare ``l2`` three ways.
#: ``losses.diffusion.lambda_mse`` is step 3 of ``_resolve_diffusion_weight``
#: and holds the *diffusion-term* weight; the table's ``lambda_mse`` -> ``l2``
#: alias is what joins it to the image ``mse`` entry.
#: ``TRIPLE_SOURCE_L2`` reduced to the pin and its alias partner, and the same block
#: with every value at the schema default. Used by ``TestPinnedLambdaIsExempt``.
PINNED_L2 = {
    "reconstruction": {"lambda_l2": 1.0},
    "diffusion": {"lambda_mse": 1.0},
    "image_losses": [{"name": "mse", "weight": 1.0, "enabled": True}],
    "policy": {"output_domain": "image"},
}
UNPINNED_L2 = {
    "reconstruction": {"lambda_l2": 0.0},
    "diffusion": {"lambda_mse": 0.0},
    "image_losses": [{"name": "mse", "weight": 0.0, "enabled": True}],
    "policy": {"output_domain": "image"},
}

TRIPLE_SOURCE_L2 = {
    "reconstruction": {"lambda_l2": 1.0, "lambda_perceptual": 0.0},
    "diffusion": {"lambda_mse": 1.0},
    "image_losses": [{"name": "mse", "weight": 1.0, "enabled": True}],
    "kspace_losses": [{"name": "complex_l1", "weight": 1.0, "enabled": True}],
    "complex_losses": [],
    "policy": {"output_domain": "kspace"},
}


class TestComputerResolvedLambdaIsExempt:
    def test_diffusion_lambda_mse_alone_is_not_flagged(self):
        """Deleting it would remove the diffusion-term knob, not a duplicate."""
        block = {
            "diffusion": {"lambda_mse": 1.0},
            "image_losses": [{"name": "mse", "weight": 1.0, "enabled": True}],
            "policy": {"output_domain": "image"},
        }
        assert _failures(block) == []

    def test_the_exemption_is_narrow(self):
        """A plain reconstruction lambda beside the same list entry still fires."""
        block = {
            "reconstruction": {"lambda_l2": 1.0},
            "image_losses": [{"name": "mse", "weight": 1.0, "enabled": True}],
            "policy": {"output_domain": "image"},
        }
        assert len(_failures(block)) == 1

    def test_triple_source_reports_neither_lambda(self):
        """``experiment_130_ti_ccd``: both lambdas stay, for unrelated reasons.

        ``losses.diffusion.lambda_mse`` is the computer-resolved knob;
        ``losses.reconstruction.lambda_l2`` is the pin holding that knob and the
        list entry at one value. An earlier revision of this check reported the
        second, the migration deleted it, and the two arms carrying this block
        turned red in ``tests/unit/config/test_exp11_kspace_filling_loss_weights``.
        """
        assert _failures(TRIPLE_SOURCE_L2) == []

    def test_exempt_source_still_conflicts_on_disagreement(self):
        """The exemption suppresses the *duplication* report, never the raise.

        ``build_loss_weight_table`` still aliases ``lambda_mse`` to ``l2``, so a
        diffusion knob turned off beside a live image ``mse`` term is rejected at
        load time. Pinned here because it is a real configuration the corpus
        cannot express, not a property this check may quietly change.
        """
        from spectramr.domain.exceptions import ConfigurationError
        from spectramr.models.losses.weights import build_loss_weight_table

        block = {
            "diffusion": {"lambda_mse": 0.0},
            "image_losses": [{"name": "mse", "weight": 1.0, "enabled": True}],
            "policy": {"output_domain": "image"},
        }
        with pytest.raises(ConfigurationError, match="DIFFERENT weights"):
            build_loss_weight_table(_settings(block).losses)

        # The audit surfaces that raise as an error result rather than aborting
        # the run, so the exemption cannot turn a hard conflict into silence.
        failures = _failures(block)
        assert [r.severity for r in failures] == ["error"]


class TestPairingRuleHasOneOwner:
    def test_check_consumes_the_shared_helper(self):
        """Two copies of the rule would let the migration outrun the audit (#17)."""
        import inspect

        from spectramr.infrastructure.validation import config_health_checker as mod

        src = inspect.getsource(mod.ConfigHealthChecker.check_loss_weight_declaration_ssot)
        assert "dual_surface_loss_declarations" in src
        assert "spec.source.split" not in src

    def test_helper_returns_the_sites_the_message_reports(self):
        from spectramr.infrastructure.validation.config_health_checker import (
            dual_surface_loss_declarations,
        )

        pairs = dual_surface_loss_declarations(_settings(EXP11_LOSSES_BEFORE).losses)
        assert {name for name, _, _, _ in pairs} == set(EXP11_DUPLICATES)
        for name, _, lambda_sites, list_sites in pairs:
            assert lambda_sites and list_sites
            # The section the migration will edit is the one the fixture records.
            assert lambda_sites[0].rsplit(".", 1)[0] == EXP11_DUPLICATES[name]


class TestPinnedLambdaIsExempt:
    """The pin rule and the shape it must NOT swallow (non-negotiable 15).

    ``PINNED_L2`` and ``UNPINNED_L2`` differ only in the declared value. ``reconstruction.lambda_l2`` defaults to 0.0 and ``diffusion.lambda_mse``
    to 1.0 (issue #421), so a declared 1.0 is the only thing holding the two equal
    and a declared 0.0 holds nothing.
    """

    def test_a_pin_is_not_reported(self):
        assert _failures(PINNED_L2) == []

    def test_a_lambda_equal_to_its_default_is_still_reported(self):
        """Deleting this one moves nothing, so the migration must still be told."""
        failures = _failures(UNPINNED_L2)
        assert len(failures) == 1
        assert "losses.reconstruction.lambda_l2" in failures[0].message

    def test_the_weight_owner_agrees_with_the_check(self):
        """One owner decides what a pin is; the audit only asks it (#17)."""
        from spectramr.models.losses.weights import deleting_lambda_would_conflict

        assert deleting_lambda_would_conflict(
            _settings(PINNED_L2).losses, "reconstruction", "lambda_l2"
        )
        assert not deleting_lambda_would_conflict(
            _settings(UNPINNED_L2).losses, "reconstruction", "lambda_l2"
        )

    def test_an_unparseable_source_segment_is_not_read_as_permission(self):
        """A shape the parser cannot read must not authorise a deletion (#9)."""
        from spectramr.infrastructure.validation.config_health_checker import (
            _pins_materialised_agreement,
        )

        losses = _settings(UNPINNED_L2).losses
        assert _pins_materialised_agreement(losses, "losses.reconstruction.lambda_l2") is False
        for shape in ("lambda_l2", "losses.lambda_l2", "a.b.c.d", "training.recon.lambda_l2"):
            assert _pins_materialised_agreement(losses, shape) is True
