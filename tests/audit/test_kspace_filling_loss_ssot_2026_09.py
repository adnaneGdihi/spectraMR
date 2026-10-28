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


#: Arms that list ``mse`` in a domain list and once also set ``losses.diffusion.lambda_mse``.
_MSE_ARMS = (
    "experiment_130_ti_ccd.yaml",
    "experiment_cross_contrast_kspace_diffusion.yaml",
    "experiment_cross_contrast_kspace_diffusion_noprior_baseline.yaml",
    "experiment_cross_contrast_kspace_diffusion_noprior_flair.yaml",
    "experiment_cross_contrast_kspace_diffusion_refenc.yaml",
    "experiment_cross_contrast_kspace_diffusion_t2_to_flair.yaml",
)


def _cohort_arm(name):
    (path,) = [arm for arm in _ARMS if arm.name == name]
    return path


def _raw_diffusion_weight(config):
    """The diffusion-term weight exactly as the strategy's computer resolves it."""
    from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
        UnifiedDiffusionLossComputer,
    )

    computer = UnifiedDiffusionLossComputer.__new__(UnifiedDiffusionLossComputer)
    computer.config = config
    computer._scheduled_override = lambda _name: None
    return computer._resolve_diffusion_weight()


class TestWhatTheMigrationDeliberatelyLeft:
    """Two shapes that look like the violation and are not.

    Pinned so a later "tidy-up" cannot delete them under this rule's banner.
    """

    def test_the_mse_arms_train_the_diffusion_mse_once(self):
        """The six MSE arms list ``mse`` in a domain list and carry no ``losses.diffusion``.

        ``losses.diffusion.lambda_mse: 1.0`` was read raw by
        ``_resolve_diffusion_weight`` ahead of ``training.diffusion.lambda_mse: 0.0``,
        so these arms trained an MSE twice: once as the strategy's diffusion term
        and once as the list entry. With the block gone the raw
        reader falls through to the training knob, and the list entry is the
        single declaration.
        """
        for name in _MSE_ARMS:
            config = _load(_cohort_arm(name))
            assert config.losses.diffusion is None, name
            listed = {
                c.name
                for domain in ("kspace_losses", "image_losses", "complex_losses", "latent_losses")
                for c in getattr(config.losses, domain)
            }
            assert "mse" in listed, name
            assert _raw_diffusion_weight(config) == 0.0, name

    def test_a_restored_diffusion_block_turns_the_raw_weight_back_on(self, tmp_path):
        """Planted: the deleted key, put back, is what the guard above must see."""
        import yaml

        raw = yaml.safe_load(_cohort_arm(_MSE_ARMS[0]).read_text())
        raw["losses"]["diffusion"] = {"lambda_mse": 1.0}
        planted_path = tmp_path / _MSE_ARMS[0]
        planted_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        planted = _load(planted_path)
        assert planted.losses.diffusion is not None
        assert _raw_diffusion_weight(planted) == 1.0

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
