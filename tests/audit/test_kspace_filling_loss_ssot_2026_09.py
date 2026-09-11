"""One declaration surface per loss weight, across the whole kspace_filling cohort.

The cohort declared the same five k-space weights twice: once as
``losses.<section>.lambda_<name>`` and once as an entry in ``kspace_losses``. Only
the list entry builds the loss module and selects the FFT bridge, so the lambda
beside it built nothing while remaining free to drift away from the weight that
actually trains. The cohort was migrated 2026-09-07 (836d29785) by
``scripts/migrations/migrate_loss_lambdas_to_domain_lists.py``. What it left behind
falls in two classes, each pinned below rather than counted here — a count in
prose drifts on its own: ``TestWhatTheMigrationDeliberatelyLeft`` holds the shapes
that only *look* like the violation, and ``KEPT_BY_MIGRATION_DENY`` holds the one
arm where the violation is real and the remedy is not yet safe.

This is the ratchet that keeps them migrated. It is deliberately broader than
``test_kspace_filling_cohort_invariants.py``, which excludes the ``ablations*``
directories by construction: the duplication was uniform across the cohort, so
the guard has to be too.

Scope note: the corpus is enumerated through :mod:`tests.utils.corpus`, whose
docstring records why an on-disk ``rglob`` gives a different subject on every
machine. An untracked arm is invisible here by design.
"""

from __future__ import annotations

import pytest

from tests.utils.corpus import repo_root, tracked_yamls

_COHORT = repo_root() / "experiments" / "inprogress" / "kspace_filling"
_ARMS = tracked_yamls(_COHORT)

#: Arms whose duplicate pair is REAL and whose deletion is not yet safe, as
#: ``{arm filename: loss name}``.
#:
#: **Empty, and that is the finished state, not an unwritten one.** The single
#: entry this list was created for — ``experiment_11_kspace_cold_diffusion_perceptual.yaml``
#: / ``perceptual`` — has been discharged, and both halves of the reason it
#: existed are gone:
#:
#: * the *mechanism* was retired by the change that made ``LossBuilder`` resolve
#:   the composite-GAN perceptual weight through the loss-weight table instead of
#:   reading ``losses.reconstruction.lambda_perceptual`` raw, so a deleted lambda
#:   no longer falls back to the schema default of ``10.0``; and
#: * the *arm* was then swept, dropping its defensive ``lambda_perceptual: 0.1``
#:   and leaving ``image_losses[perceptual].weight`` as the single declaration.
#:
#: Verified after the sweep: the arm still resolves ``perceptual`` at ``0.1``,
#: sourced ``losses.image_losses[perceptual].weight``. The effective weight did
#: not move — one surface now says what two used to.
#: ``scripts/migrations/migrate_loss_lambdas_to_domain_lists.py`` retired its
#: matching ``DENY`` entry and its ``_raw_reader_census`` pin in the same change.
#:
#: The mechanism stays for the next genuine exemption. While the list is empty
#: ``TestTheKeptPairIsStillReal`` parametrizes over nothing, which pytest reports
#: as ``SKIPPED (got empty parameter set)`` — visible, and the correct resting
#: state. Do not replace that with an ``assert not KEPT_BY_MIGRATION_DENY``:
#: forbidding the list from ever being used again is a different rule than the
#: one this file enforces, and it would fire on a legitimate future entry.
KEPT_BY_MIGRATION_DENY: dict[str, str] = {}


def _pairs(losses):
    from spectramr.infrastructure.validation.config_health_checker import (
        dual_surface_loss_declarations,
    )

    return dual_surface_loss_declarations(losses)


def _load(path):
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.from_yaml(str(path))


class TestTheCohortDeclaresEachWeightOnce:
    @pytest.mark.parametrize("arm", _ARMS, ids=lambda p: p.name)
    def test_no_arm_declares_a_weight_on_both_surfaces(self, arm):
        kept = KEPT_BY_MIGRATION_DENY.get(arm.name)
        pairs = [p for p in _pairs(_load(arm).losses) if p[0] != kept]
        assert pairs == [], "\n".join(
            f"{name} (weight={weight}) declared at {lam} AND {lst}"
            for name, weight, lam, lst in pairs
        )

    def test_the_scan_is_not_vacuous(self):
        """A cohort that enumerates empty would pass every assertion above.

        Pinned as a floor, not an equality: arms are added to this cohort often,
        and a ratchet that fails on a new arm teaches people to delete the
        ratchet. 50 is comfortably below the 59 present at the migration.
        """
        assert len(_ARMS) >= 50, f"only {len(_ARMS)} tracked arms found under {_COHORT}"


class TestTheGuardCanFail:
    """A gate is only a gate for a violation shape it has been watched to reject.

    The corpus is clean, so the assertions above pass whether or not the
    predicate works. These plant the two shapes the migration removed.
    """

    def _settings(self, losses_block):
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

    def test_a_reconstruction_lambda_beside_its_list_entry_is_caught(self):
        """The shape 52 arms carried, five times over."""
        pairs = _pairs(
            self._settings(
                {
                    "reconstruction": {"lambda_complex_l1": 1.0},
                    "kspace_losses": [{"name": "complex_l1", "weight": 1.0, "enabled": True}],
                    "policy": {"output_domain": "kspace"},
                }
            ).losses
        )
        assert [p[0] for p in pairs] == ["complex_l1"]

    def test_a_physics_lambda_beside_its_list_entry_is_caught(self):
        """``complex_spatial_gradient`` lives in a different schema class."""
        pairs = _pairs(
            self._settings(
                {
                    "physics": {"lambda_complex_spatial_gradient": 1.0},
                    "kspace_losses": [
                        {
                            "name": "complex_spatial_gradient",
                            "weight": 1.0,
                            "enabled": True,
                        }
                    ],
                    "policy": {"output_domain": "kspace"},
                }
            ).losses
        )
        assert [p[0] for p in pairs] == ["complex_spatial_gradient"]


class TestWhatTheMigrationDeliberatelyLeft:
    """Two shapes that look like the violation and are not.

    Pinned so a later "tidy-up" cannot delete them under this rule's banner.
    """

    def test_the_two_triple_source_arms_keep_their_diffusion_knob(self):
        """``losses.diffusion.lambda_mse`` is the diffusion-term weight.

        A computer reads it directly (step 3 of ``_resolve_diffusion_weight``);
        the weight table merely aliases ``lambda_mse`` onto ``l2``, which is what
        joins it to the ``mse`` list entry. Deleting it removes a knob.
        """
        arms = [
            _COHORT / "experiment_130_ti_ccd.yaml",
            _COHORT / "experiment_cross_contrast_kspace_diffusion.yaml",
        ]
        for arm in arms:
            losses = _load(arm).losses
            assert losses.diffusion is not None
            assert "lambda_mse" in losses.diffusion.model_fields_set
            assert _pairs(losses) == []

    def test_the_two_triple_source_arms_keep_their_pinned_l2(self):
        """``reconstruction.lambda_l2`` is what holds that knob and the list entry
        at one value.

        Its own schema default is 0.0 against ``diffusion.lambda_mse``'s 1.0
        (issue #421), so deleting it makes the two disagree. An earlier revision
        of the migration deleted it from exactly these two arms and
        ``tests/unit/config/test_exp11_kspace_filling_loss_weights.py`` went red;
        both arms are therefore unchanged by this PR.
        """
        from spectramr.models.losses.weights import deleting_lambda_would_conflict

        for name in (
            "experiment_130_ti_ccd.yaml",
            "experiment_cross_contrast_kspace_diffusion.yaml",
        ):
            losses = _load(_COHORT / name).losses
            assert "lambda_l2" in losses.reconstruction.model_fields_set
            assert deleting_lambda_would_conflict(losses, "reconstruction", "lambda_l2")

    def test_lambda_pre_dc_kspace_survives_where_it_is_declared(self):
        """It has no list form at all — the diffusion strategy reads it inline.

        Its schema default is 0.0, so deleting it would switch off a term the arm
        switches on. At least one migrated arm must still carry it.
        """
        carriers = [
            arm
            for arm in _ARMS
            if (recon := _load(arm).losses.reconstruction) is not None
            and "lambda_pre_dc_kspace" in recon.model_fields_set
        ]
        assert carriers, "no arm declares lambda_pre_dc_kspace; the term went missing"


class TestTheKeptPairIsStillReal:
    """``KEPT_BY_MIGRATION_DENY`` retires itself when its reason expires.

    An exception list is the standard way a ratchet rots: the entry outlives the
    hazard, and the arm stays exempt for a reason nobody re-reads. So both facts
    the exemption rests on are asserted here rather than described in a comment.
    Either one going false turns this class red, and the failure is the
    instruction to delete the entry — never to widen it.
    """

    @pytest.mark.parametrize("arm_name,loss", sorted(KEPT_BY_MIGRATION_DENY.items()))
    def test_the_exempted_arm_still_carries_the_pair(self, arm_name, loss):
        """Red once the arm is swept, which is when the entry must go.

        This is what makes the exemption temporary. It also catches the quieter
        failure: an entry naming an arm or a loss that no longer exists silences
        the ratchet for a violation that could reappear under that name.
        """
        arm = next((a for a in _ARMS if a.name == arm_name), None)
        assert arm is not None, (
            f"{arm_name} is exempted in KEPT_BY_MIGRATION_DENY but is not a tracked "
            f"arm of this cohort — delete the entry"
        )
        assert loss in [name for name, _, _, _ in _pairs(_load(arm).losses)], (
            f"{arm_name} no longer declares {loss} on both surfaces — the exemption "
            f"is spent. Delete its entry from KEPT_BY_MIGRATION_DENY."
        )

    @pytest.mark.parametrize("arm_name,loss", sorted(KEPT_BY_MIGRATION_DENY.items()))
    def test_deleting_the_lambda_would_not_be_a_no_op(self, arm_name, loss):
        """The justifying predicate: the lambda is holding back the schema default.

        Deleting a lambda is safe exactly when the field falls back to the value
        it already has. Here it does not — the arm declares 0.1 against a default
        of 10.0 — so the deletion the ratchet would demand is a hundredfold
        change to what ``LossBuilder`` reads raw, not a tidy-up. Asserted as a
        *difference* rather than against the literal 10.0: if #1923 changes the
        default, the hazard changes shape and this must be re-derived, not
        silently re-confirmed.
        """
        from spectramr.config.schemas.loss import ReconstructionLossesConfig

        arm = next(a for a in _ARMS if a.name == arm_name)
        recon = _load(arm).losses.reconstruction
        field = f"lambda_{loss}"
        assert recon is not None and field in recon.model_fields_set, (
            f"{arm_name} does not declare {field} explicitly; the exemption "
            f"describes a lambda that is not there"
        )
        assert getattr(recon, field) != getattr(ReconstructionLossesConfig(), field), (
            f"{arm_name}'s {field} now equals the schema default, so deleting it "
            f"changes nothing. The exemption is spent — delete its entry."
        )
