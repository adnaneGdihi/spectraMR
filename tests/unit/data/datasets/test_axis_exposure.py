"""Tests for the dataset_type axis / spatial-rank exposure tables.

The centrepiece is :class:`TestCanonicalKeyContract`, which asserts every key of
both tables is a *canonical* ``dataset_type``. It drives the real
``DataConfigSchema`` validator rather than a copy of its valid/alias lists: a
copied list rots independently and would keep passing while the real validator
disagrees, which is the exact bug the guard exists to catch.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.config.schemas.enums import Axis
from spectramr.data.datasets.axis_exposure import (
    _BART_ROLE_TO_AXIS,
    _BART_ROLES_WITHOUT_AXIS,
    DATASET_TYPE_AXES,
    DATASET_TYPE_RANKS,
    DATASET_TYPE_SIGNAL_DOMAINS,
    declared_axes_for,
    exposed_axes_for,
    resolve_axes_for,
    signal_domains_for,
    spatial_rank_for,
)


def _normalize(raw: str) -> str:
    """Return what ``data.dataset_type: <raw>`` becomes after real validation.

    Calls the schema's own ``dataset_type`` field validator, which is the real
    lower-case + alias-rewrite + reject seam that produces the value the health
    checker later reads. Raises for a value the schema rejects.

    Deliberately NOT a copy of the valid/alias lists: a copied list rots
    independently of the validator and would keep passing while the real one
    disagrees, which is the bug class this module's guard exists to catch.

    Deliberately NOT full ``DataConfigSchema(...)`` construction either: the
    model carries cross-field validators unrelated to the canonical-key question
    (``contrast_aware_paired`` additionally demands ``input_contrast`` /
    ``target_contrast``), so constructing would conflate "this key is not
    canonical" with "this data block is incomplete".
    :meth:`test_field_validator_agrees_with_model_construction` pins this call to
    the real construction path.
    """
    return DataConfigSchema.validate_dataset_type(raw)


ALL_KEYS = sorted(
    {*DATASET_TYPE_AXES, *DATASET_TYPE_RANKS, *DATASET_TYPE_SIGNAL_DOMAINS}
)


class TestCanonicalKeyContract:
    """Every table key must survive validation unchanged, or it is a dead rule."""

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_key_is_a_canonical_dataset_type(self, key: str) -> None:
        # An invalid key raises here (dead: the config never reaches the check).
        normalized = _normalize(key)
        # An alias key normalizes to something else (dead: the lookup sees the
        # canonical value, never the alias).
        assert normalized == key, (
            f"{key!r} is not canonical: validation rewrites it to {normalized!r}, "
            f"so the table entry can never be hit. Key the table on {normalized!r}."
        )

    @pytest.mark.parametrize("raw", ["2d", "3d", "kspace", "m4raw", "cine", "IMAGE"])
    def test_field_validator_agrees_with_model_construction(self, raw: str) -> None:
        # Binds _normalize's direct field-validator call to the real validation
        # path, so the shortcut cannot drift from what a loaded config actually
        # holds. Restricted to types that construct without cross-field deps.
        assert _normalize(raw) == DataConfigSchema(dataset_type=raw).dataset_type

    def test_guard_rejects_an_alias_key(self) -> None:
        # Proves the guard above has teeth: '2d' was a real key until it was
        # found dead, and it must never come back.
        assert _normalize("2d") == "image" != "2d"

    def test_guard_rejects_an_invalid_key(self) -> None:
        # 'slice' was a real key and is not even a valid dataset_type, so such a
        # config dies at Tier 0 and never reaches the workflow check.
        with pytest.raises(ValueError, match="'slice' is not recognised"):
            _normalize("slice")

    def test_deleted_dead_keys_are_absent(self) -> None:
        for dead in (
            "2d",
            "3d",
            "paired_nifti",
            "paired_mri",
            "slice",
            "volumetric",
            "reconstruction",
        ):
            assert dead not in DATASET_TYPE_AXES
        for dead in ("2d", "3d", "slice", "volumetric"):
            assert dead not in DATASET_TYPE_RANKS


class TestTableValueTypes:
    def test_axes_values_are_frozensets_of_axis(self) -> None:
        for key, axes in DATASET_TYPE_AXES.items():
            assert isinstance(axes, frozenset), key
            assert all(isinstance(a, Axis) for a in axes), key

    def test_rank_values_are_positive_ints(self) -> None:
        for key, rank in DATASET_TYPE_RANKS.items():
            assert isinstance(rank, int), key
            assert rank in (2, 3), key


class TestExposedAxesFor:
    def test_annotated_canonical_key_returns_its_axes(self) -> None:
        assert exposed_axes_for("cine") == frozenset({Axis.TEMPORAL})

    def test_annotated_canonical_key_may_expose_nothing(self) -> None:
        # frozenset() is a real annotation ("carries no non-spatial axis") and
        # must stay distinct from None ("unannotated, skip").
        assert exposed_axes_for("image") == frozenset()
        assert exposed_axes_for("image") is not None

    def test_unannotated_canonical_key_returns_none(self) -> None:
        # 'synthetic' is a valid dataset_type left deliberately unannotated, so
        # the required-axes rule skips it rather than guessing.
        assert _normalize("synthetic") == "synthetic"
        assert exposed_axes_for("synthetic") is None

    def test_alias_key_returns_none(self) -> None:
        # Callers must pass post-validation values. '2d' is unannotated here by
        # construction; the annotation lives on its canonical target 'image'.
        assert exposed_axes_for("2d") is None
        assert exposed_axes_for(_normalize("2d")) == frozenset()

    def test_none_returns_none(self) -> None:
        assert exposed_axes_for(None) is None


class TestSpatialRankFor:
    def test_annotated_canonical_key_returns_its_rank(self) -> None:
        assert spatial_rank_for("npy_slice") == 2

    def test_unannotated_canonical_key_returns_none(self) -> None:
        # 'nifti' is annotated for axes but deliberately NOT for rank: the index
        # builder emits per-slice records (rank 2) under variant '2d_slices' and
        # whole volumes (rank 3) otherwise, so the rank is not a property of the
        # dataset_type alone.
        assert _normalize("nifti") == "nifti"
        assert spatial_rank_for("nifti") is None
        assert exposed_axes_for("nifti") is not None

    def test_alias_key_returns_none(self) -> None:
        assert spatial_rank_for("3d") is None
        assert _normalize("3d") == "nifti"

    def test_none_returns_none(self) -> None:
        assert spatial_rank_for(None) is None


class TestClusterDataAnnotations:
    """The two dataset_types that read this cluster's data, plus their neighbours."""

    def test_m4raw_exposes_coil(self) -> None:
        # M4Raw 0.3T is 4-coil raw k-space.
        assert exposed_axes_for("m4raw") == frozenset({Axis.COIL})
        assert spatial_rank_for("m4raw") == 2

    def test_m4raw_does_not_expose_temporal(self) -> None:
        # Regression: the NEX repetitions are phase-incoherent re-acquisitions of
        # a STATIC object, not a temporal axis. Claiming TEMPORAL would let a
        # dynamic/functional/perfusion arm pass the axis guard on static 0.3T
        # brain data (pitfall #19) and would have temporal_fidelity grade the rep
        # mean (which IS the target) as a failure.
        axes = exposed_axes_for("m4raw")
        assert axes is not None
        assert Axis.TEMPORAL not in axes
        assert Axis.ECHO not in axes

    def test_kspace_exposes_no_axis(self) -> None:
        # Static single-contrast fastMRI k-space. COIL is withheld because the
        # single-coil layout has no coil axis.
        assert exposed_axes_for("kspace") == frozenset()
        assert Axis.COIL not in exposed_axes_for("kspace")

    def test_fastmri_aliases_all_land_on_the_kspace_annotation(self) -> None:
        for alias in ("fastmri_kspace", "fastmri_knee", "fastmri_brain", "volume_h5"):
            assert _normalize(alias) == "kspace"
            assert exposed_axes_for(_normalize(alias)) == frozenset()

    def test_mrixfields_exposes_no_axis(self) -> None:
        # Magnitude-only paired multi-field translation; field strength is not an
        # Axis member and contrast is a Task.
        assert exposed_axes_for("mrixfields") == frozenset()

    def test_mrixfields_rank_is_omitted_because_a_serving_knob_decides_it(self) -> None:
        # mrixfields was rank 2 until MRIxFieldsPairedDataset gained a
        # mrixfields_slice_mode knob (central/all_slices -> 2, volume -> 3). Like
        # nifti, a knob this table cannot see now decides the served rank, so the
        # key is omitted and the rank check SKIPS rather than pinning a wrong value.
        assert "mrixfields" not in DATASET_TYPE_RANKS
        assert spatial_rank_for("mrixfields") is None


class TestRequiredAxesEnforcement:
    """The tables must actually reject the regimes they exist to reject."""

    def test_only_phase_contrast_types_expose_velocity_encoding(self) -> None:
        # A flow arm on M4Raw (or any magnitude corpus) must still be rejected; only
        # the phase-contrast loaders carry a velocity-encoding axis.
        exposing = {t for t, axes in DATASET_TYPE_AXES.items() if Axis.VELOCITY_ENCODING in axes}
        assert exposing == {"flow_synthetic"}

    @pytest.mark.parametrize(
        "required",
        [
            Axis.TEMPORAL,
            Axis.ECHO,
            Axis.SPECTRAL,
            Axis.DIFFUSION_ENCODING,
            Axis.TRANSIENT,
        ],
    )
    def test_m4raw_misses_every_regime_axis(self, required: Axis) -> None:
        # Each of these is some regime's required_axes; m4raw must satisfy none,
        # which is what makes check_workflow_required_axes fire on this cluster.
        exposed = exposed_axes_for("m4raw")
        assert exposed is not None
        assert required not in exposed


class TestSignalDomains:
    """The per-dataset signal-domain SSOT: what each loader actually materialises."""

    def test_values_are_frozensets_of_known_domain_literals(self) -> None:
        from typing import get_args

        from spectramr.models.capabilities import Domain

        known = set(get_args(Domain))
        for dataset_type, domains in DATASET_TYPE_SIGNAL_DOMAINS.items():
            assert isinstance(domains, frozenset)
            assert domains, f"{dataset_type!r} has an empty signal-domain set"
            unknown = set(domains) - known
            assert not unknown, (
                f"{dataset_type!r} names signal domains {sorted(unknown)} that are "
                f"not Domain literals {sorted(known)} — no model could advertise them."
            )

    def test_mrixfields_is_image_only(self) -> None:
        # THE fix: magnitude-only MNI images. NOT kspace, NOT complex_image — a
        # k-space model pointed here reads a representation this loader never
        # produces, and there is no honest k-space of a magnitude image.
        assert signal_domains_for("mrixfields") == frozenset({"image"})
        assert "kspace" not in signal_domains_for("mrixfields")

    def test_raw_kspace_datasets_are_kspace_and_complex(self) -> None:
        for dataset_type in ("m4raw", "kspace", "bart_kspace", "ismrmrd_kspace"):
            domains = signal_domains_for(dataset_type)
            assert domains == frozenset({"kspace", "complex_image"}), dataset_type
            assert "image" not in domains

    def test_unannotated_returns_none_and_none_returns_none(self) -> None:
        # Absent = skip, the same non-breaking invariant as the axis/rank tables.
        assert signal_domains_for("synthetic") is None
        assert signal_domains_for(None) is None

    def test_alias_key_returns_none(self) -> None:
        # A pre-validation alias cannot reach a lookup here (the check reads the
        # canonical value), so an alias is unannotated by construction.
        assert signal_domains_for("fastmri_brain") is None


# The two real ``bart_dim_map`` shapes in experiments/inprogress/vf/. Copied as
# literals on purpose: if an arm's map changes, these keep testing the adapter
# rather than silently re-testing whatever the corpus now says.
_ECHO_ARM_DIM_MAP = {"readout": 1, "spoke": 2, "coil": 3, "echo": 6, "slice": 10, "repetition": 13}
_FLIP_ARM_DIM_MAP = {"readout": 0, "phase": 1, "coil": 3, "flip": 11}


def _bart_data(dim_map: dict[str, int], *, enabled: bool = True) -> DataConfigSchema:
    """A REAL validated config, not a SimpleNamespace stub.

    Hand-rolled stubs are how a block-reading check has previously reported "no
    such section" and PASSED; ``declared_axes_for`` reads ``data.bart``, so the
    thing under test is exactly the attribute a stub would fake into existence.
    """
    return DataConfigSchema(
        dataset_type="bart_kspace",
        bart={
            "enabled": enabled,
            "bart_dim_map": dim_map,
            "sampling": "radial",
            "trajectory_source": "golden_angle",
        },
    )


class TestBartRoleAdapterCoverage:
    """The two BART/Axis vocabularies meet in exactly one place, and it is total."""

    def test_every_bart_role_is_mapped_or_explicitly_excused(self) -> None:
        """A role added to _BART_DIM_ROLES later cannot map to nothing in silence.

        This is the guard against the adapter-bypass failure mode: without it, a
        new role simply produces no axis, the arm's exposure quietly shrinks, and
        every existing test still passes.
        """
        from spectramr.config.schemas.data import _BART_DIM_ROLES

        covered = set(_BART_ROLE_TO_AXIS) | set(_BART_ROLES_WITHOUT_AXIS)
        assert covered == set(_BART_DIM_ROLES), (
            "every BART dim role must be either mapped to an Axis or listed in "
            "_BART_ROLES_WITHOUT_AXIS with a reason. Unhandled: "
            f"{sorted(set(_BART_DIM_ROLES) - covered)}; stale: "
            f"{sorted(covered - set(_BART_DIM_ROLES))}"
        )

    def test_the_two_tables_are_disjoint(self) -> None:
        assert not set(_BART_ROLE_TO_AXIS) & set(_BART_ROLES_WITHOUT_AXIS)

    def test_mapped_values_are_real_axis_members(self) -> None:
        for role, axis in _BART_ROLE_TO_AXIS.items():
            assert isinstance(axis, Axis), f"{role!r} maps to a non-Axis {axis!r}"

    def test_every_exclusion_carries_a_reason(self) -> None:
        for role, reason in _BART_ROLES_WITHOUT_AXIS.items():
            assert reason.strip(), f"{role!r} is excused without a reason"

    def test_repetition_is_not_temporal(self) -> None:
        """NEX averages of a static object are not a time series.

        The same call ``DATASET_TYPE_AXES['m4raw']`` makes. Mapping it would let a
        dynamic / functional / perfusion arm declare its regime on static data and
        pass, which is the pitfall-#19 hole this whole table exists to close.
        """
        assert "repetition" not in _BART_ROLE_TO_AXIS
        assert Axis.TEMPORAL not in declared_axes_for(_bart_data(_ECHO_ARM_DIM_MAP))

    def test_spatial_roles_are_never_axes(self) -> None:
        """``Axis`` is by construction the NON-spatial vocabulary."""
        for role in ("readout", "phase", "phase2", "spoke", "slice"):
            assert role in _BART_ROLES_WITHOUT_AXIS


class TestDeclaredAxesFor:
    """A per-arm declaration, read from the config the schema already validated."""

    def test_echo_arm_declares_echo_and_coil(self) -> None:
        assert declared_axes_for(_bart_data(_ECHO_ARM_DIM_MAP)) == frozenset({Axis.ECHO, Axis.COIL})

    def test_flip_arm_declares_flip_angle_and_coil(self) -> None:
        """``flip`` maps to ``Axis.FLIP_ANGLE`` as of #1020/#1025.

        It previously had no ``Axis`` member, so this asserted coil only and was
        named ``..._declares_coil_only`` for that absence. B1+ transmit-field
        mapping varies the flip angle the way a multi-echo arm varies TE, so the
        axis is first-class now and the arm declares it rather than dropping it.
        """
        assert declared_axes_for(_bart_data(_FLIP_ARM_DIM_MAP)) == frozenset(
            {Axis.FLIP_ANGLE, Axis.COIL}
        )

    def test_disabled_bart_declares_nothing(self) -> None:
        assert declared_axes_for(_bart_data(_ECHO_ARM_DIM_MAP, enabled=False)) is None

    def test_config_without_a_bart_block_declares_nothing(self) -> None:
        assert declared_axes_for(DataConfigSchema(dataset_type="image")) is None

    def test_object_without_the_attribute_declares_nothing(self) -> None:
        """The checker's existing SimpleNamespace call sites must keep falling back."""
        from types import SimpleNamespace

        assert declared_axes_for(SimpleNamespace(dataset_type="image")) is None
        assert declared_axes_for(None) is None

    def test_spatial_only_declaration_is_an_empty_set_not_a_skip(self) -> None:
        """A positive "this arm carries no non-spatial axis", which must REJECT.

        ``None`` and ``frozenset()`` are the load-bearing distinction in this
        module: the first skips the rule, the second fails it. An arm that
        declares only spatial roles has told us enough to reject a temporal
        regime, so collapsing the two would silently restore the skip.
        """
        declared = declared_axes_for(_bart_data({"readout": 0, "phase": 1}))
        assert declared == frozenset()
        assert declared is not None


class TestTheDeclaredRouteCannotBeSilentlyDisabled:
    """``declared_axes_for`` reads ``data.bart`` by ``getattr``, which is quiet.

    The duck-typing is deliberate — it is what lets the checker's existing
    SimpleNamespace call sites fall back to the type table instead of crashing.
    But quiet is two-sided: rename or remove the schema field and the declared
    route stops resolving with no error anywhere, and every test above still
    passes because they would all take the same ``None`` branch.

    So pin the field's existence against the REAL schema. A rename then reds this
    test rather than silently reverting eight arms to "skipped". Same guard shape
    as the ``_BART_DIM_ROLES`` coverage test above, and the failure it prevents is
    the one already loose in the codebase: a config stand-in that has drifted from
    its schema does not announce itself.
    """

    def test_data_schema_still_has_the_bart_field(self) -> None:
        assert "bart" in DataConfigSchema.model_fields, (
            "declared_axes_for resolves data.bart by getattr; if the field moved "
            "or was renamed, the declared-axis route is now inert and every "
            "bart_kspace arm has silently gone back to skipping "
            "check_workflow_required_axes."
        )

    def test_the_bart_block_still_has_the_dim_map(self) -> None:
        from spectramr.config.schemas.data import BartConfigSchema

        for field in ("enabled", "bart_dim_map"):
            assert field in BartConfigSchema.model_fields, (
                f"declared_axes_for reads bart.{field} by getattr; a rename "
                "makes the declared route resolve to None in silence."
            )


class TestBartKspaceIsDeliberatelyUnannotated:
    def test_bart_kspace_has_no_static_row(self) -> None:
        """Its axes are a per-ARM fact, so no per-TYPE annotation can be right.

        Five bart arms declare ``echo``, three declare ``flip``. Any single
        annotation would be wrong for one group — which is precisely why the
        declared route exists rather than a new table row.
        """
        assert "bart_kspace" not in DATASET_TYPE_AXES
        assert exposed_axes_for("bart_kspace") is None


class TestResolveAxesFor:
    """One resolver for "declared wins", shared by the audit and the batch.

    The composition was inlined in ``check_workflow_required_axes`` while it had
    a single consumer. ``TrainingBatch.axes`` is the second, and two copies of a
    precedence rule is how the audit and the tensor start disagreeing about the
    same arm -- the divergent-sibling shape this module documents for
    ``bart_kspace``.
    """

    @staticmethod
    def _cfg(dataset_type=None, dim_map=None, bart_enabled=False):
        import types

        bart = types.SimpleNamespace(enabled=bart_enabled, bart_dim_map=dim_map or {})
        return types.SimpleNamespace(dataset_type=dataset_type, bart=bart)

    def test_declaration_outranks_the_type_annotation(self) -> None:
        """An arm that declares echo beats a dataset_type annotated as empty."""
        from spectramr.config.schemas.enums import Axis
        from spectramr.data.datasets.axis_exposure import resolve_axes_for

        cfg = self._cfg(dataset_type="image", dim_map={"echo": 6}, bart_enabled=True)
        assert resolve_axes_for(cfg) == frozenset({Axis.ECHO})

    def test_falls_back_to_the_annotation_when_nothing_is_declared(self) -> None:
        from spectramr.config.schemas.enums import Axis
        from spectramr.data.datasets.axis_exposure import resolve_axes_for

        assert resolve_axes_for(self._cfg(dataset_type="cine")) == frozenset(
            {Axis.TEMPORAL}
        )
        assert resolve_axes_for(self._cfg(dataset_type="m4raw")) == frozenset(
            {Axis.COIL}
        )

    def test_unannotated_type_with_no_declaration_is_none_not_empty(self) -> None:
        """The load-bearing distinction. ``None`` = cannot vouch, SKIP.
        ``frozenset()`` = positively no non-spatial axis, REJECT a regime that
        needs one. Collapsing them would silently turn every unannotated arm
        into a claim that it carries nothing.
        """
        from spectramr.data.datasets.axis_exposure import resolve_axes_for

        assert resolve_axes_for(self._cfg(dataset_type="synthetic")) is None
        assert resolve_axes_for(self._cfg(dataset_type=None)) is None

    def test_an_annotated_type_with_no_axes_is_empty_not_none(self) -> None:
        from spectramr.data.datasets.axis_exposure import resolve_axes_for

        assert resolve_axes_for(self._cfg(dataset_type="image")) == frozenset()

    def test_it_agrees_with_the_two_routes_it_composes(self) -> None:
        """Parametrised over the whole table rather than one example, so a new
        row cannot land with the composition disagreeing about it."""
        from spectramr.data.datasets.axis_exposure import (
            DATASET_TYPE_AXES,
            exposed_axes_for,
            resolve_axes_for,
        )

        for dataset_type in DATASET_TYPE_AXES:
            cfg = self._cfg(dataset_type=dataset_type)
            assert resolve_axes_for(cfg) == exposed_axes_for(dataset_type)

    def test_the_health_checker_uses_this_resolver_not_its_own_copy(self) -> None:
        """The point of extracting it. A re-inlined composition reads the same."""
        import inspect

        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        src = inspect.getsource(ConfigHealthChecker.check_workflow_required_axes)
        assert "resolve_axes_for(data_cfg)" in src
        assert "else exposed_axes_for(" not in src, (
            "the declared-wins composition was re-inlined; it must have one home"
        )


# ---------------------------------------------------------------------------
# coil_processing_mode overrides the dataset_type row (#1010)
#
# `check_workflow_dataset_signal_domain` read DATASET_TYPE_SIGNAL_DOMAINS by
# dataset_type alone and reported 16 arms across 10 model families as "the model
# cannot read what this dataset produces". Every one of the 16 sets
# `coil_processing_mode: rss_image`, which applies the IFFT INSIDE the dataset's
# TorchIO transform pipeline -- so they are served images, exactly as their
# image-domain models expect. The arms were right; the check was reading a row
# that no longer described them.
#
# Precisely the oversight `needs_ifft_for_visualization` had until 2026-05-15,
# when ignoring the same knob IFFT'd already-image tensors and produced the
# spectral/tiled-noise aliasing in ten smoke fakes.
# ---------------------------------------------------------------------------


class TestCoilModeOverridesTheTypeRow:
    @staticmethod
    def _cfg(mode: str | None):
        from spectramr.config.schemas.data import DataConfigSchema

        kw = {"dataset_type": "kspace"}
        if mode is not None:
            kw["coils"] = {"processing_mode": mode}
        return DataConfigSchema(**kw)

    def test_an_image_mode_makes_a_kspace_arm_serve_images(self) -> None:
        from spectramr.data.datasets.axis_exposure import resolve_signal_domains_for

        assert resolve_signal_domains_for(self._cfg("rss_image")) == frozenset({"image"})
        assert resolve_signal_domains_for(self._cfg("magnitude")) == frozenset({"image"})

    def test_a_non_image_mode_leaves_the_type_row_intact(self) -> None:
        """Anti-vacuity: the override must not swallow the check.

        A genuine kspace-out arm feeding an image model is still a real mismatch,
        and this is the case that proves the fix did not just disarm the rule.
        """
        from spectramr.data.datasets.axis_exposure import (
            resolve_signal_domains_for,
            signal_domains_for,
        )

        by_type = signal_domains_for("kspace")
        assert by_type == frozenset({"complex_image", "kspace"}), "precondition"
        assert resolve_signal_domains_for(self._cfg("none")) == by_type
        assert resolve_signal_domains_for(self._cfg(None)) == by_type

    def test_the_mode_set_has_one_definition(self) -> None:
        """Two hand-maintained copies is how audit and runtime diverge."""
        from spectramr.data.datasets import axis_exposure as data_side
        from spectramr.infrastructure.training.utils import domain_inference as infra_side

        assert (
            data_side.IMAGE_DOMAIN_COIL_MODES is infra_side.IMAGE_DOMAIN_COIL_MODES
        )

    def test_the_check_passes_the_arms_it_used_to_reject(self) -> None:
        """End to end on a real arm, which is what #1010 was actually about."""
        import copy
        from pathlib import Path

        import yaml

        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        repo = Path(__file__).resolve().parents[4]
        arm = (
            repo
            / "experiments/inprogress/diffusion/baseline_m4raw_ddpm_rss.yaml"
        )
        if not arm.exists():  # pragma: no cover - corpus arm may be renamed
            pytest.skip(f"corpus arm absent: {arm}")
        raw = yaml.safe_load(arm.read_text())
        raw["workflow"] = {"regime": "mri_structural"}

        checker = ConfigHealthChecker()
        good = checker.check_workflow_dataset_signal_domain(
            TrainingSettings.settings_from_dict(copy.deepcopy(raw))
        )
        assert good.passed, good.message
        assert "coil_processing_mode" in good.message, (
            "the pass must say WHY the domain resolved to image"
        )

        # Same arm, coil combination switched off: the mismatch is real again.
        raw["data"].setdefault("coils", {})["processing_mode"] = "none"
        bad = checker.check_workflow_dataset_signal_domain(
            TrainingSettings.settings_from_dict(raw)
        )
        assert not bad.passed, "the check must still fire on a genuine mismatch"


# ---------------------------------------------------------------------------
# The temporal route, and the per-arm declaration (#998)
#
# Seven of nine live regimes were declarable on ZERO arms in the corpus, because
# `mri_structural` is the only one requiring no non-spatial axis and no reachable
# loader exposed `temporal` / `transient` / `echo`. The first instinct -- annotate
# `nifti` with TEMPORAL -- is precisely what this module's contract forbids: a
# 4-D BOLD series read as `nifti` is folded into channels with the frame order,
# count and TR dropped, so the claim would be false and would let a functional
# arm pass on mangled data.
#
# Two honest fixes instead: a loader that actually keeps the time axis legible
# (`fmri`), and a per-arm declaration for facts no per-type row can carry.
# ---------------------------------------------------------------------------


class TestTemporalRouteIsReachable:
    def test_fmri_exposes_temporal_and_nifti_still_does_not(self) -> None:
        """The pair is the point: the annotation tracks the LOADER, not the file.

        Both read 4-D NIfTI. Only one keeps the axis legible, and only that one
        may claim it.
        """
        assert exposed_axes_for("fmri") == frozenset({Axis.TEMPORAL})
        assert exposed_axes_for("nifti") == frozenset()

    def test_fmri_is_a_selectable_dataset_type(self) -> None:
        """An annotation on a dataset_type no config can name would be inert."""
        from spectramr.config.schemas.data import CANONICAL_DATASET_TYPES

        assert "fmri" in CANONICAL_DATASET_TYPES

    def test_fmri_is_not_rank_annotated(self) -> None:
        """Deliberate: 2-D+t and 3-D+t both route here, so no rank is honest.

        Same argument that omits ``nifti`` and ``mrixfields``. Pinning rank 3
        would reject a legitimate 2-D+t arm.
        """
        assert "fmri" not in DATASET_TYPE_RANKS

    def test_the_loader_refuses_a_three_d_volume(self) -> None:
        """What makes the TEMPORAL claim true per-sample rather than on average."""
        import numpy as np
        import pytest

        from spectramr.data.datasets.fmri_dataset import (
            FMRIBoldSeriesDataset,
            build_fmri_index,
        )

        tmp = Path(tempfile.mkdtemp())
        np.save(tmp / "vol.npy", np.zeros((4, 4, 2), dtype="float32"))
        # The rank check precedes the target lookup, so a declared pairing with
        # no sibling on disk still exercises it.
        ds = FMRIBoldSeriesDataset(
            build_fmri_index(tmp, "**/*.npy"), target_source="sibling"
        )
        with pytest.raises(ValueError, match="4-D BOLD series"):
            _ = ds[0]

    def test_the_loader_preserves_what_makes_the_axis_readable(self) -> None:
        """Frame order / count / TR are the difference from the `nifti` route."""
        import numpy as np

        from spectramr.data.datasets.fmri_dataset import (
            FMRIBoldSeriesDataset,
            build_fmri_index,
        )

        tmp = Path(tempfile.mkdtemp())
        np.save(tmp / "bold.npy", np.zeros((5, 6, 2, 3), dtype="float32"))  # H,W,D,T
        # A paired target is now mandatory (#739): input == target is the
        # degenerate case both fMRI strategies solve trivially.
        np.save(tmp / "bold_target.npy", np.ones((5, 6, 2, 3), dtype="float32"))
        subject = FMRIBoldSeriesDataset(
            build_fmri_index(tmp, "**/bold.npy"),
            tr_seconds=1.5,
            target_source="sibling",
        )[0]

        assert tuple(subject["input"].data.shape) == (3, 5, 6, 2), "T must lead"
        assert subject["num_frames"] == 3
        assert subject["frame_order"] == [0, 1, 2]
        assert subject["tr"] == 1.5


class TestPerArmAxisDeclaration:
    @staticmethod
    def _cfg(**kw):
        from spectramr.config.schemas.data import DataConfigSchema

        return DataConfigSchema(dataset_type=kw.pop("dataset_type", "quantitative"), **kw)

    def test_an_undeclared_arm_falls_back_to_the_table(self) -> None:
        """``None`` must stay "no declaration", not "declares nothing"."""
        assert declared_axes_for(self._cfg()) is None
        assert resolve_axes_for(self._cfg(dataset_type="m4raw")) == frozenset(
            {Axis.COIL}
        )

    def test_a_declaration_outranks_the_type_annotation(self) -> None:
        """The whole point: a per-arm fact beats a per-corpus generalisation."""
        cfg = self._cfg(dataset_type="m4raw", acquisition_axes=["echo"])
        assert resolve_axes_for(cfg) == frozenset({Axis.ECHO})

    def test_an_empty_list_is_a_positive_claim_not_a_skip(self) -> None:
        """``[]`` must REJECT a regime requiring an axis; ``None`` must skip.

        Collapsing the two is the distinction the whole optional-set design
        exists to preserve.
        """
        assert declared_axes_for(self._cfg(acquisition_axes=[])) == frozenset()
        assert declared_axes_for(self._cfg()) is None

    def test_a_typo_raises_at_the_schema(self) -> None:
        """Silently dropping an unknown name would invert the declaration.

        ``["echoes"]`` resolving to ``frozenset()`` would reject the very arm the
        author was trying to admit.
        """
        import pytest

        with pytest.raises(ValueError, match="is not an Axis"):
            self._cfg(acquisition_axes=["echoes"])

    def test_declarations_union_with_the_bart_dim_map(self) -> None:
        """Both are positive statements about one acquisition, so neither wins.

        A BART arm that also declares an axis its dim map cannot express should
        keep both facts.
        """
        cfg = self._cfg(
            dataset_type="bart_kspace",
            acquisition_axes=["temporal"],
            bart={"enabled": True, "bart_dim_map": {"echo": 5}},
        )
        assert declared_axes_for(cfg) == frozenset({Axis.TEMPORAL, Axis.ECHO})

    def test_it_unblocks_a_regime_that_had_no_declarable_arm(self) -> None:
        """End to end: the audit check is what this exists to satisfy."""
        from spectramr.config.schemas.enums import Regime
        from spectramr.domain.workflows import WORKFLOW_PROFILES

        required = WORKFLOW_PROFILES[Regime.QUANTITATIVE].required_axes
        # ANY of the required axes satisfies the regime -- mirror
        # ``check_workflow_required_axes``, which tests ``required & exposed``.
        # This line read ``required <= resolved``, an all-of test that agreed
        # with the checker only while every profile declared a single axis. Once
        # QUANTITATIVE came to mean "echo OR flip_angle" (#1020) the subset form
        # became a SECOND, STRICTER encoding of the rule: it would demand an arm
        # expose echo AND flip_angle, which no B1+ or multi-echo arm ever does.
        #
        # The precondition is a membership test, not an equality pin. Pinning the
        # whole set is what went stale here, and it would go stale again the next
        # time the regime gains an axis; ``Axis.ECHO in required`` is the only
        # fact this test needs -- it is what makes the echo arm's intersection
        # non-empty.
        assert Axis.ECHO in required, "precondition"
        resolved = resolve_axes_for(self._cfg(acquisition_axes=["echo"])) or frozenset()
        assert required & resolved


# ---------------------------------------------------------------------------
# The fmri route must not clone input into target (#739)
#
# The route added with `dataset_type: fmri` emitted `target = input.clone()` --
# FieldRefDataset's "degradation twin" pattern, which is legitimate there
# because a degradation transform sits downstream. Nothing degrades a BOLD
# series, so the twin is the degenerate case both fMRI strategies solve
# trivially:
#
#   BeltramiEPIDistortion:  residual = L1(apply_epi_distortion(target, dB0), input)
#                           with input == target, dB0 = 0 zeroes residual AND
#                           mu_reg simultaneously -> field map is worthless.
#   SpatiotemporalAdaptiveSFCRecon: L1(G(input), target) -> the identity.
#
# Both train smoothly to a worthless answer with every metric improving. The
# repo's own backlog called this out before the route existed
# (TODO/inprogress/backlog_fmri_serving_path_2026_08_05.md #2); the route shipped
# anyway. The pairing is now declared or the dataset refuses to build.
# ---------------------------------------------------------------------------


class TestFmriPairingMustBeDeclared:
    @staticmethod
    def _volume(tmp, name: str):
        import numpy as np

        np.save(tmp / name, np.random.rand(6, 6, 3, 4).astype("float32"))

    def test_an_undeclared_pairing_refuses_to_build(self) -> None:
        import tempfile
        from pathlib import Path

        from spectramr.data.datasets.fmri_dataset import (
            FMRIBoldSeriesDataset,
            build_fmri_index,
        )

        tmp = Path(tempfile.mkdtemp())
        self._volume(tmp, "sub-01_bold.npy")
        with pytest.raises(ValueError, match="target_source"):
            FMRIBoldSeriesDataset(build_fmri_index(tmp, "**/*.npy"))

    def test_a_missing_sibling_refuses_rather_than_cloning(self) -> None:
        """The failure mode that matters: silence here trains a trivial model."""
        import tempfile
        from pathlib import Path

        from spectramr.data.datasets.fmri_dataset import (
            FMRIBoldSeriesDataset,
            build_fmri_index,
        )

        tmp = Path(tempfile.mkdtemp())
        self._volume(tmp, "sub-01_bold.npy")
        ds = FMRIBoldSeriesDataset(
            build_fmri_index(tmp, "**/*.npy"), target_source="sibling"
        )
        with pytest.raises(FileNotFoundError, match="no sibling target"):
            _ = ds[0]

    def test_a_paired_acquisition_yields_a_distinct_target(self) -> None:
        """Anti-vacuity: the whole point is that target is not input."""
        import tempfile
        from pathlib import Path

        import torch

        from spectramr.data.datasets.fmri_dataset import (
            FMRIBoldSeriesDataset,
            build_fmri_index,
        )

        tmp = Path(tempfile.mkdtemp())
        self._volume(tmp, "sub-01_bold.npy")
        self._volume(tmp, "sub-01_bold_target.npy")
        subject = FMRIBoldSeriesDataset(
            build_fmri_index(tmp, "**/sub-01_bold.npy"), target_source="sibling"
        )[0]
        assert not torch.equal(subject["input"].data, subject["target"].data), (
            "target must be a genuinely different acquisition"
        )

    def test_the_schema_offers_no_self_pairing_option(self) -> None:
        """A 'self' escape hatch would reintroduce the degenerate twin."""
        from spectramr.config.schemas.data import FmriConfigSchema

        with pytest.raises(ValueError):
            FmriConfigSchema(enabled=True, target_source="self")


class TestTheCoilModeReadHasOneSpelling:
    """`_coil_processing_mode` reads only `coils.processing_mode`.

    It carried a fallback onto the flat `coil_processing_mode`, which is a FOLD
    record -- so the attribute does not exist on a loaded config and the branch
    returned None for every arm. It read canonical-first, so nothing was broken,
    but it re-reddened `test_no_unallowlisted_string_keyed_reads` hours after
    #1016 cleared that gate. Landed in 907b827d3 (#1010).
    """

    @staticmethod
    def _schema(mode: str | None):
        from spectramr.config.schemas.data import DataConfigSchema

        kw: dict = {"dataset_type": "kspace"}
        if mode is not None:
            kw["coils"] = {"processing_mode": mode}
        return DataConfigSchema(**kw)

    def test_the_canonical_path_is_read(self) -> None:
        from spectramr.data.datasets.axis_exposure import _coil_processing_mode

        assert _coil_processing_mode(self._schema("rss_image")) == "rss_image"

    def test_the_legacy_flat_spelling_folds_rather_than_being_read(self) -> None:
        """Declaring the legacy name still works -- via the FOLD, not via a
        second read. This is why deleting the fallback changes no behaviour."""
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.data.datasets.axis_exposure import _coil_processing_mode

        cfg = DataConfigSchema(dataset_type="kspace", coil_processing_mode="rss_image")
        assert cfg.coils.processing_mode == "rss_image"
        assert not hasattr(cfg, "coil_processing_mode"), (
            "the flat attribute must not survive the fold -- if it does, the "
            "deleted fallback was live after all"
        )
        assert _coil_processing_mode(cfg) == "rss_image"

    def test_a_real_config_never_reaches_the_absent_branch(self) -> None:
        """`coils` has a default_factory and `processing_mode` defaults to the
        real mode ``'none'`` — so an arm that declares nothing still resolves to
        a DECLARED value, not to the empty string. That distinction is why
        deleting the flat fallback is safe: there was never a loaded config for
        which the canonical read came back empty and the fallback could fire.
        """
        from spectramr.data.datasets.axis_exposure import _coil_processing_mode

        assert self._schema(None).coils.processing_mode == "none"
        assert _coil_processing_mode(self._schema(None)) == "none"

    def test_only_a_non_schema_object_yields_the_empty_string(self) -> None:
        """Anti-vacuity: the guard must still tolerate an object with no block,
        or `resolve_signal_domains_for` raises instead of falling through."""
        import types

        from spectramr.data.datasets.axis_exposure import _coil_processing_mode

        assert _coil_processing_mode(types.SimpleNamespace()) == ""
