"""Unit tests for DataConfigSchema loader-knob handling (WS4).

Pins the ``max_prefetch`` → ``prefetch_factor`` fold: ``prefetch_factor`` is the
single authoritative knob every live loader path reads, and the deprecated
``max_prefetch`` alias is folded into it non-destructively so pre-existing YAMLs
(8 arms set ``max_prefetch``) keep both their value and their ability to load
under ``extra="forbid"``.
"""

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.data import (
    DataConfigSchema,
    DataLoaderConfigSchema,
    MRIxFieldsDataConfigSchema,
)


class TestPrefetchFold:
    """Both legacy spellings land on ``data.loader.prefetch_factor`` (phase 9a).

    ``max_prefetch`` was already a deprecated alias resolved by a hand-written
    validator; it is now a rename record like any other, so the shim, the fixer
    and the corpus gate read one table instead of three.
    """

    def test_legacy_max_prefetch_folds_into_prefetch_factor(self):
        cfg = DataConfigSchema(max_prefetch=4)
        assert cfg.loader.prefetch_factor == 4

    def test_legacy_prefetch_factor_folds_too(self):
        cfg = DataConfigSchema(prefetch_factor=5)
        assert cfg.loader.prefetch_factor == 5

    def test_declaring_both_spellings_at_different_values_raises(self):
        """Renamed and inverted from ``test_explicit_prefetch_factor_wins_over_max_prefetch``.

        That test asserted a precedence rule -- explicit beats alias -- which
        only existed because the two names were separate fields. They now fold
        onto ONE field, so a document setting both is a document that disagrees
        with itself, and picking a winner would silently discard the other. The
        author is asked instead.

        Renamed rather than flipped in place: a test whose name asserts a
        precedence it no longer checks is worse than no test.
        """
        with pytest.raises(ValidationError, match="disagree"):
            DataConfigSchema(max_prefetch=4, prefetch_factor=6)

    def test_both_spellings_agreeing_is_accepted(self):
        """Agreement is not a conflict -- the duplicate is simply dropped."""
        cfg = DataConfigSchema(max_prefetch=4, prefetch_factor=4)
        assert cfg.loader.prefetch_factor == 4

    def test_neither_set_uses_default(self):
        assert DataConfigSchema().loader.prefetch_factor == 2

    def test_max_prefetch_key_still_accepted(self):
        # The 8 legacy YAMLs must still load: the key folds, it is not rejected.
        cfg = DataConfigSchema(max_prefetch=3)
        assert cfg.loader.prefetch_factor == 3

    def test_use_async_dataloader_still_accepted(self):
        cfg = DataConfigSchema(use_async_dataloader=False)
        assert cfg.use_async_dataloader is False


class TestTargetMode:
    """M4Raw NEX averaging knob — phase_aligned_mean fixes complex-mean cancellation."""

    def test_default_is_complex_mean(self):
        assert DataConfigSchema().target_mode == "complex_mean"

    def test_phase_aligned_mean_accepted(self):
        assert (
            DataConfigSchema(target_mode="phase_aligned_mean").target_mode == "phase_aligned_mean"
        )

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValidationError):
            DataConfigSchema(target_mode="bogus")

    def test_nex_target_exclude_input_defaults_false(self):
        # Default off -> byte-identical legacy all-reps target.
        assert DataConfigSchema().nex_target_exclude_input is False

    def test_nex_target_exclude_input_opt_in(self):
        assert DataConfigSchema(nex_target_exclude_input=True).nex_target_exclude_input is True

    # --- use_repetitions: the route decides (2026-09-02, cohort review T0.1) ---

    def test_use_repetitions_defaults_to_none_so_the_route_decides(self):
        """None, not False: the m4raw route averages repetitions by default and
        46 corpus arms never declare the knob. A False default wired as-is would
        have flipped them to single-rep targets."""
        assert DataConfigSchema().use_repetitions is None

    @pytest.mark.parametrize("declared", [True, False])
    def test_use_repetitions_explicit_value_is_kept(self, declared):
        assert DataConfigSchema(use_repetitions=declared).use_repetitions is declared

    # --- nex_fallback: what a <3-rep group does under leave-one-out ---

    def test_nex_fallback_defaults_to_error(self):
        assert DataConfigSchema().nex_fallback == "error"

    def test_nex_fallback_all_reps_requires_leave_one_out(self):
        """The knob is read only under leave-one-out; declaring it with the
        all-reps reference is an unread knob (non-negotiable 8) -> refuse."""
        with pytest.raises(ValidationError, match="nex_fallback is read only"):
            DataConfigSchema(nex_fallback="all_reps")

    def test_nex_fallback_all_reps_accepted_under_leave_one_out(self):
        cfg = DataConfigSchema(nex_target_exclude_input=True, nex_fallback="all_reps")
        assert cfg.nex_fallback == "all_reps"

    def test_nex_fallback_unknown_value_rejected(self):
        with pytest.raises(ValidationError):
            DataConfigSchema(nex_target_exclude_input=True, nex_fallback="warn")

    def test_max_prefetch_respects_ge_2(self):
        with pytest.raises(ValidationError):
            DataConfigSchema(max_prefetch=1)


class TestBidirectionalMode:
    """ULF↔HF direction knob; hf_to_hf / ulf_to_ulf are single-field autoencode."""

    def test_default_is_ulf_to_hf(self):
        assert DataConfigSchema().pairing.bidirectional_mode == "ulf_to_hf"

    @pytest.mark.parametrize("mode", ["ulf_to_hf", "hf_to_ulf", "hf_to_hf", "ulf_to_ulf"])
    def test_valid_modes_accepted(self, mode):
        assert DataConfigSchema(bidirectional_mode=mode).pairing.bidirectional_mode == mode

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValidationError):
            DataConfigSchema(bidirectional_mode="hf_to_ulf_wrong")


def test_mrixfields_rescale_per_image_defaults_off() -> None:
    """The MRIxFields corpus is already [0,1]; per-image renorm is opt-in.

    Renormalising source and target each to [0,1] erases the cross-field intensity
    relationship that field translation exists to learn, so the default must stay
    off. Pinned here because flipping it silently changes the reference every
    MRIxFields metric grades against.
    """
    from spectramr.config.schemas.data import DataConfigSchema

    assert MRIxFieldsDataConfigSchema.model_fields["rescale_per_image"].default is False
    cfg = DataConfigSchema(data_dir="/tmp", dataset_type="mrixfields")
    assert cfg.mrixfields.rescale_per_image is False
    assert (
        DataConfigSchema(
            data_dir="/tmp",
            dataset_type="mrixfields",
            mrixfields={"rescale_per_image": True},
        ).mrixfields.rescale_per_image
        is True
    )


class TestKSpaceScaleDomain:
    """``kspace_scale_domain`` selects where the robust scale is measured.

    Added when the double k-space normalization (dataset AND transform) was
    removed: the Parseval/image-domain scale that ``M4RawRepetitionDataset``
    computed inline moved into ``KSpaceNormalizationTransform``, so arms that
    need it must be able to ask for it.
    """

    def test_defaults_to_kspace(self):
        assert DataConfigSchema().processing.kspace_scale_domain == "kspace"

    def test_accepts_image(self):
        assert (
            DataConfigSchema(
                processing={"kspace_scale_domain": "image"}
            ).processing.kspace_scale_domain
            == "image"
        )

    def test_rejects_unknown_domain(self):
        """Closed enum — a typo must fail at load time, not degrade silently."""
        with pytest.raises(ValidationError):
            DataConfigSchema(processing={"kspace_scale_domain": "parseval"})


class TestNormalizationTypeLiteral:
    """The Literal may only advertise strategies the transform builder implements.

    ``'scalar'`` was a member with NO dispatch branch in either
    ``build_train_transforms`` or ``build_val_transforms``: an arm declaring it
    loaded clean, warned once on the train side, appended nothing, and then
    trained AND was graded on un-normalised data. Removed 2026-08-04 (0 arms
    declared it) -- pitfall #15, an advertised knob nothing reads. Re-add it
    here only together with both dispatch branches.
    """

    _SUPPORTED = ("none", "standard", "minmax", "percentile", "robust_percentile")

    def test_scalar_is_rejected_by_the_sub_block(self):
        from spectramr.config.schemas.data import DataProcessingConfigSchema

        with pytest.raises(ValidationError, match="normalization_type"):
            DataProcessingConfigSchema(normalization_type="scalar")

    def test_scalar_is_rejected_through_the_legacy_flat_spelling_too(self):
        """The fold must not smuggle a retired value past the Literal."""
        with pytest.raises(ValidationError, match="normalization_type"):
            DataConfigSchema(normalization_type="scalar")

    @pytest.mark.parametrize("value", _SUPPORTED)
    def test_supported_values_validate(self, value):
        from spectramr.config.schemas.data import DataProcessingConfigSchema

        assert DataProcessingConfigSchema(normalization_type=value).normalization_type == value
        assert DataConfigSchema(normalization_type=value).processing.normalization_type == value

    def test_robust_percentile_is_still_accepted(self):
        """It has no dispatch branch either, but unlike ``'scalar'`` it FOLDS.

        ``TorchIOTransformConfig.from_training_config`` rewrites it to
        ``'percentile'`` before the dispatch sees it, so the value is wired --
        which is exactly the difference that decided which of the two members
        was removed. The fold itself is pinned in
        ``tests/unit/data/builders/test_torchio_transform_builder.py::
        TestNormalizationTypeSchemaBuilderBridge``.
        """
        assert (
            DataConfigSchema(normalization_type="robust_percentile").processing.normalization_type
            == "robust_percentile"
        )

    def test_default_is_none(self):
        assert DataConfigSchema().processing.normalization_type == "none"


class TestNoUndeclaredDataBlockReads:
    """No code may read a ``config.data.<x>`` that ``DataConfigSchema`` does not declare.

    ``DataConfigSchema`` is ``extra="ignore"``, which hides this two ways:

    * a bare ``config.data.foo`` raises ``AttributeError`` **at runtime**, on a
      path no test covers — this is how ``make_dataset`` / ``make_dataloader``
      crashed on their first line for every config, and how ``infer``'s
      preprocessing crashed on its gate;
    * a ``hasattr(config.data, "foo")`` guard is permanently ``False``, so the
      branch behind it is dead and the mechanism it gates never fires
      (pitfall #9 / #16) — this is how ``multislice_enabled`` was advertised in
      two docstrings while being unreachable.

    AST-based, not a grep: source text cannot distinguish a real attribute read
    from the same words in a docstring or a log message, and it would flag
    method calls such as ``config.data.resolve_mode(...)``.

    Verified to fail on the pre-fix tree with 6 names across 14 sites.
    """

    _CONFIG_CHAINS = ("config.data", "settings.data")

    @staticmethod
    def _dotted(node) -> str | None:
        """Flatten ``a.b.c`` to ``"a.b.c"``; ``None`` if the base is not a Name."""
        import ast

        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            return None
        parts.append(cur.id)
        return ".".join(reversed(parts))

    @classmethod
    def _aliases_in(cls, scope) -> set[str]:
        """Locals bound to the data block by assignments directly in ``scope``.

        Two forms, both real in-tree: ``x = config.data`` and
        ``x = getattr(config, "data", None)`` (the latter is what
        ``pipelines/train.py`` used, which hid its dead ``image_size`` branch).

        The source must be a *config* chain, not merely ``.data``: torch tensors
        and TorchIO subjects both carry ``.data``, so a bare ``attr == "data"``
        rule collects every ``x = subject[k].data`` and then reports ``.dtype``
        and ``.clone`` as undeclared config fields.
        """
        import ast

        out: set[str] = set()
        for node in ast.walk(scope):
            if not isinstance(node, ast.Assign):
                continue
            v = node.value
            ok = False
            if isinstance(v, ast.Attribute) and v.attr == "data":
                chain = cls._dotted(v)
                ok = chain is not None and chain.endswith(cls._CONFIG_CHAINS)
            elif (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Name)
                and v.func.id == "getattr"
                and len(v.args) >= 2
                and isinstance(v.args[1], ast.Constant)
                and v.args[1].value == "data"
            ):
                base = v.args[0]
                name = (
                    base.id
                    if isinstance(base, ast.Name)
                    else (cls._dotted(base) or "").rpartition(".")[2]
                )
                ok = name in {"config", "settings"}
            if ok:
                out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        return out

    def test_no_undeclared_config_data_attribute_reads(self) -> None:
        import ast
        import pathlib

        from spectramr.config.schemas.data import DataConfigSchema

        # Methods / properties / validators are legitimate non-field attributes.
        allowed = set(DataConfigSchema.model_fields) | {
            n for n in dir(DataConfigSchema) if not n.startswith("__")
        }

        offenders: dict[str, list[str]] = {}
        for path in sorted(pathlib.Path("src/spectramr").rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(errors="ignore"))
            except SyntaxError:  # pragma: no cover - not expected in-tree
                continue
            # Aliases resolve per *function*, not per module: spec_card.py binds
            # `data = getattr(config, "data", None)` in one function and uses an
            # unrelated dict also named `data` in another, so a module-wide set
            # would report that dict's `.get` as a config field.
            # Pass 1 walks the module with NO aliases (pure `config.data.x`
            # chains); pass 2 walks each function with only that function's own
            # aliases. A single module-wide alias set would let a binding in one
            # function leak into another — `spec_card._derive_data_form` binds
            # `data = getattr(config, "data", None)` while `_format_card` takes
            # an unrelated `data: dict`, whose `.get` would then read as a field.
            scopes: list = [(tree, set())] + [
                (n, self._aliases_in(n))
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for scope, aliases in scopes:
                if not aliases and scope is not tree:
                    continue
                for node in ast.walk(scope):
                    if not isinstance(node, ast.Attribute):
                        continue
                    dotted = self._dotted(node)
                    if dotted is None:
                        continue
                    head, _, leaf = dotted.rpartition(".")
                    if not (head.endswith(self._CONFIG_CHAINS) or head in aliases):
                        continue
                    if leaf in allowed:
                        continue
                    offenders.setdefault(leaf, []).append(f"{path}:{node.lineno}")

        assert not offenders, (
            "these names are read off `config.data` but DataConfigSchema does "
            "not declare them. Because the block is extra='ignore' a bare read "
            "raises AttributeError at runtime and a hasattr guard is silently "
            "False forever. Declare the field, or read the canonical one:\n  "
            + "\n  ".join(f"{k}: {sorted(set(v))}" for k, v in sorted(offenders.items()))
        )

    def test_multislice_enabled_is_declared_and_defaults_off(self) -> None:
        """Regression: two strategies documented this flag as 'an explicit config
        flag to avoid ambiguous shape heuristics' while it was undeclared, so the
        multi-slice branch could never run."""
        from spectramr.config.schemas.data import DataConfigSchema

        assert "multislice_enabled" in DataConfigSchema.model_fields
        assert DataConfigSchema().multislice_enabled is False


class TestTransformSpecSchema:
    """``data.processing.transforms`` entries are typed (finding D9).

    The field was ``list[dict[str, Any]]`` -- anything validated, and the only
    consumer matched one literal name and dropped the rest in silence.
    """

    def test_name_is_required(self):
        """One committed arm spells the key ``type:``; it must not validate."""
        from pydantic import ValidationError

        from spectramr.config.schemas.data import TransformSpecSchema

        with pytest.raises(ValidationError):
            TransformSpecSchema(type="scout_acquisition")

    def test_nested_kwargs_are_returned(self):
        from spectramr.config.schemas.data import TransformSpecSchema

        spec = TransformSpecSchema(name="phase_residual", kwargs={"kernel_size": 9})
        assert spec.resolved_kwargs() == {"kernel_size": 9}

    def test_flat_kwargs_are_accepted(self):
        """Back-compat: the committed graph_encoding arms write kwargs flat."""
        from spectramr.config.schemas.data import TransformSpecSchema

        spec = TransformSpecSchema(name="graph_encoding", k_neighbors=8, max_nodes=64)
        assert spec.resolved_kwargs() == {"k_neighbors": 8, "max_nodes": 64}

    def test_nested_kwargs_win_over_a_flat_collision(self):
        from spectramr.config.schemas.data import TransformSpecSchema

        spec = TransformSpecSchema(name="phase_residual", kernel_size=3, kwargs={"kernel_size": 11})
        assert spec.resolved_kwargs() == {"kernel_size": 11}

    def test_processing_block_accepts_a_list_of_specs(self):
        from spectramr.config.schemas.data import DataProcessingConfigSchema

        cfg = DataProcessingConfigSchema(
            transforms=[{"name": "phase_residual", "kwargs": {"kernel_size": 5}}]
        )
        assert cfg.transforms[0].name == "phase_residual"
        assert cfg.transforms[0].resolved_kwargs() == {"kernel_size": 5}


class TestDatasetTypeVocabulary:
    """``CANONICAL_DATASET_TYPES`` / ``DATASET_TYPE_ALIASES`` are the SSOT."""

    def test_aliases_all_resolve_to_a_canonical_type(self):
        from spectramr.config.schemas.data import (
            CANONICAL_DATASET_TYPES,
            DATASET_TYPE_ALIASES,
        )

        for alias, target in DATASET_TYPE_ALIASES.items():
            assert target in CANONICAL_DATASET_TYPES, f"{alias} -> {target} not canonical"

    def test_no_alias_shadows_a_canonical_name(self):
        """An alias that is also canonical would fold a valid type away."""
        from spectramr.config.schemas.data import (
            CANONICAL_DATASET_TYPES,
            DATASET_TYPE_ALIASES,
        )

        assert not (set(DATASET_TYPE_ALIASES) & set(CANONICAL_DATASET_TYPES))

    def test_graph_mri_is_no_longer_accepted(self):
        """Canonical with no dataset class; 0 arms declared it."""
        from pydantic import ValidationError

        from spectramr.config.schemas.data import DataConfigSchema

        with pytest.raises(ValidationError):
            DataConfigSchema(dataset_type="graph_mri")

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("fastmri_brain", "kspace"),
            ("2d", "image"),
            ("3d", "nifti"),
            ("paired_mri", "nifti_paired"),
            ("slice_paired", "npy_slice"),
        ],
    )
    def test_aliases_still_fold(self, alias, expected):
        from spectramr.config.schemas.data import DataConfigSchema

        assert DataConfigSchema(dataset_type=alias).dataset_type == expected

    def test_unknown_type_error_names_both_tables(self):
        from pydantic import ValidationError

        from spectramr.config.schemas.data import DataConfigSchema

        with pytest.raises(ValidationError) as exc:
            DataConfigSchema(dataset_type="nonsense")
        msg = str(exc.value)
        assert "Canonical types" in msg and "aliases" in msg


class TestTemporalTargetPairingIsDeclared:
    """``temporal.target_source`` must be said, never inferred.

    ``target_suffix is None`` could mean "self-paired" or "the author forgot".
    Reading it as the former on an arm with no degradation transform trains the
    identity and reports excellent PSNR (pitfall #16), so silence raises.
    """

    def test_enabled_without_target_source_raises(self):
        from pydantic import ValidationError

        from spectramr.config.schemas.data import TemporalConfigSchema

        with pytest.raises(ValidationError) as exc:
            TemporalConfigSchema(enabled=True)
        msg = str(exc.value)
        assert "target_source" in msg
        # The message has to name both ways out, not just the rule.
        assert "'sibling'" in msg and "'self'" in msg

    def test_disabled_block_is_untouched(self):
        """Every non-cine arm carries this block by default_factory.

        A required field here would have broken all of them; the guard is
        gated on ``enabled`` for that reason.
        """
        from spectramr.config.schemas.data import TemporalConfigSchema

        assert TemporalConfigSchema().target_source is None

    def test_sibling_without_suffix_raises(self):
        from pydantic import ValidationError

        from spectramr.config.schemas.data import TemporalConfigSchema

        with pytest.raises(ValidationError, match="target_suffix"):
            TemporalConfigSchema(enabled=True, target_source="sibling")

    def test_self_with_suffix_raises_as_incoherent(self):
        """A suffix that is never read is a config the author misunderstood."""
        from pydantic import ValidationError

        from spectramr.config.schemas.data import TemporalConfigSchema

        with pytest.raises(ValidationError, match="no sibling is ever read"):
            TemporalConfigSchema(enabled=True, target_source="self", target_suffix="_gt.nii.gz")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_source": "self"},
            {"target_source": "sibling", "target_suffix": "_gt.nii.gz"},
        ],
    )
    def test_coherent_declarations_construct(self, kwargs):
        from spectramr.config.schemas.data import TemporalConfigSchema

        cfg = TemporalConfigSchema(enabled=True, **kwargs)
        assert cfg.target_source == kwargs["target_source"]

    def test_glob_pattern_is_a_knob_not_a_hardcode(self):
        """It was hardcoded in ``build_cine_index``'s only caller.

        That made every non-ACDC layout -- and the ``.npy`` / ``.pt`` formats
        ``_load_4d_volume`` supports -- unreachable.
        """
        from spectramr.config.schemas.data import TemporalConfigSchema

        assert TemporalConfigSchema().glob_pattern == "**/*4d.nii.gz"
        cfg = TemporalConfigSchema(enabled=True, target_source="self", glob_pattern="**/*.npy")
        assert cfg.glob_pattern == "**/*.npy"


class TestQuantitativeInputSourceIsDeclared:
    """``quantitative.input_source`` must be stated, never inferred (2026-08-05).

    ``input_paths`` being the target maps is a legitimate mode (generative
    modelling of the map distribution — what ``build_nist_mrf_manifest.py``
    writes) and also the identity trap for any arm that means to *predict* the
    maps. The two are indistinguishable from the config alone, so the arm
    declares which it is and ``QuantitativeMapDataset`` checks the claim.
    """

    def test_enabled_without_input_source_raises(self):
        from spectramr.config.schemas.data import QuantitativeConfigSchema

        with pytest.raises(ValidationError, match="requires an explicit"):
            QuantitativeConfigSchema(enabled=True, target_maps=["t1", "t2"])

    @pytest.mark.parametrize("source", ["contrasts", "maps"])
    def test_declared_input_source_is_accepted(self, source):
        from spectramr.config.schemas.data import QuantitativeConfigSchema

        cfg = QuantitativeConfigSchema(enabled=True, target_maps=["t1", "t2"], input_source=source)
        assert cfg.input_source == source

    def test_disabled_block_needs_no_declaration(self):
        """The default (disabled) block must stay constructible — 1000+ arms
        never mention `quantitative:` at all."""
        from spectramr.config.schemas.data import QuantitativeConfigSchema

        assert QuantitativeConfigSchema().input_source is None

    def test_unknown_input_source_is_rejected(self):
        from spectramr.config.schemas.data import QuantitativeConfigSchema

        with pytest.raises(ValidationError):
            QuantitativeConfigSchema(enabled=True, target_maps=["t1"], input_source="whatever")


class TestLegacySizeKeyDoesNotFightTheDrain:
    """`image_size` must not be injected beside a drained `sampling.patch_size`.

    `migrate_legacy_sizes` maps `img_size`/`target_size`/`image_size` onto
    `patch_size`, guarded by `"patch_size" not in data`. That guard reads only
    the LEGACY spelling, and `data.patch_size` folds to
    `data.sampling.patch_size` -- so once the key drain rewrites an arm, the
    legacy key is gone, the guard fires, and a scalar `image_size` lands beside
    the arm's declared canonical list. `_reject_disagreeing_spellings` then
    refuses a config that loaded fine before a rewrite meant to be a no-op.

    Measured when it bit: 75 of 133 arms in the first three cohorts drained,
    and 97 in-progress arms across 15 cohorts declare both keys.
    """

    def test_a_drained_arm_with_a_legacy_size_key_still_loads(self):
        cfg = DataConfigSchema(image_size=256, sampling={"patch_size": [256, 256, 1]})
        assert tuple(cfg.sampling.patch_size) == (256, 256, 1)

    def test_the_same_shape_under_every_legacy_size_spelling(self):
        """All three names share the guard, so all three share the bug."""
        for legacy in ("img_size", "target_size", "image_size"):
            cfg = DataConfigSchema(**{legacy: 256}, sampling={"patch_size": [256, 256, 1]})
            assert tuple(cfg.sampling.patch_size) == (256, 256, 1), legacy

    def test_the_migration_still_happens_when_nothing_declares_patch_size(self):
        """Anti-vacuity. A guard that simply stopped migrating would satisfy
        every assertion above while silently dropping the legacy key."""
        cfg = DataConfigSchema(image_size=[128, 128, 1])
        assert tuple(cfg.sampling.patch_size) == (128, 128, 1)

    def test_the_undrained_spelling_still_wins_over_the_legacy_size_key(self):
        """Pre-drain behaviour is unchanged: an arm declaring both
        `patch_size` and `image_size` keeps `patch_size`."""
        cfg = DataConfigSchema(patch_size=[64, 64, 1], image_size=256)
        assert tuple(cfg.sampling.patch_size) == (64, 64, 1)


class TestDataOutputDomainIsDrained:
    """``data.output_domain`` is promoted to ``raise`` (Wave 3.3, #919).

    The leaf ``output_domain`` has **three unrelated owners**, which is what
    makes this the record where a leaf-name measurement does real damage:

    - ``data.output_domain`` -- this record, 0 corpus declarations.
    - ``losses.output_domain`` -- a separate fold record with **101** live
      corpus declarations, which must keep folding.
    - the ``@register_model`` ``output_domain`` capability -- 51 files under
      ``models/generators/``.

    A grep for the leaf returns 2926 hits corpus-wide and every one of them
    belongs to one of the other two. Promoting on that number would have
    promoted the wrong record and broken 101 arms at load.
    """

    def test_the_drained_flat_spelling_now_raises(self) -> None:
        with pytest.raises(ValidationError, match="output_domain"):
            DataConfigSchema(output_domain="image")

    def test_the_error_names_the_canonical_replacement(self) -> None:
        """A rename that raises without naming its destination is a dead end."""
        with pytest.raises(ValidationError) as exc:
            DataConfigSchema(output_domain="image")
        assert "data.domain.output" in str(exc.value)

    def test_the_canonical_path_still_works(self) -> None:
        """Anti-vacuity: a schema that rejected the block outright would satisfy
        the raise-test above while breaking every arm that migrated."""
        assert DataConfigSchema(domain={"output": "kspace"}).domain.output == "kspace"

    def test_the_losses_sibling_still_folds(self) -> None:
        """The asymmetry is the whole point: same leaf, different record.

        ``losses.output_domain`` has 101 live corpus declarations. If a future
        change promotes it because it shares a spelling with this one, those 101
        arms stop loading -- so pin the two postures apart here rather than
        leaving the distinction to a comment.
        """
        from spectramr.config.schemas.renames import RENAMES

        assert RENAMES["data.output_domain"].posture == "raise"
        assert RENAMES["losses.output_domain"].posture == "fold"


# ---------------------------------------------------------------------------
# num_workers is a memory multiplier, and said so nowhere
# ---------------------------------------------------------------------------
class TestNumWorkersDocumentsItsMemoryCost:
    """``num_workers`` shipped as a bare ``Field(default=4, ge=0)`` -- no
    description at all -- while its own sibling in this same module,
    ``MRIxFieldsDataConfigSchema.max_resident_volumes``, documented the very
    multiplication it participates in (``~226 MB per volume ... times
    num_workers``).

    So the downstream knob explained the multiplier and the knob that *does* the
    multiplying was silent. Anyone sizing a node read ``Field(default=4, ge=0)``
    and got no signal that raising it to 8 costs ~2.5 GB of resident memory.
    """

    def test_the_knob_is_documented(self) -> None:
        """The defect itself: a knob that multiplies memory must say so
        (non-negotiable 8 -- an exposed knob is read, validated *and* legible)."""
        desc = DataLoaderConfigSchema.model_fields["num_workers"].description
        assert desc, "num_workers multiplies per-worker memory and must document it"

    def test_it_names_pss_as_the_sizing_instrument(self) -> None:
        """Pinned on the instrument name, not the sentence around it, so a reword
        does not break this.

        PSS is load-bearing rather than decorative: summing per-worker RSS counts
        the shared torch fork image once per worker and reports 9671 MB where the
        cgroup charges 3944 MB. A description that omitted the instrument would
        leave a reader sizing from the 2.45x-overstated number -- which is the
        exact mistake the measurement behind this text was correcting.
        """
        desc = DataLoaderConfigSchema.model_fields["num_workers"].description or ""
        assert "PSS" in desc

    def test_the_assertion_is_not_vacuous(self) -> None:
        """Anti-vacuity: ``.description`` must actually be able to come back empty,
        or the two assertions above pass for reasons unrelated to this field.

        ``batch_size`` sits directly above ``num_workers`` in the same schema and
        is still bare, so it stands as the live control. If it ever gains a
        description this assertion should be repointed, not deleted -- the point
        is that *some* field proves the accessor reports absence.
        """
        assert DataLoaderConfigSchema.model_fields["batch_size"].description is None


class TestHeldOutTestIndex:
    """``data.source.test_index_path`` (cohort review 2026-09-02, T0.3)."""

    def test_defaults_to_none(self):
        from spectramr.config.schemas.data import DataSourceConfigSchema

        assert DataSourceConfigSchema().test_index_path is None

    @pytest.mark.parametrize("twin", ["index_path", "validation_index_path"])
    def test_test_index_equal_to_train_or_val_is_rejected(self, twin):
        """The planted violation: a 'held-out' set that is the training set."""
        from spectramr.config.schemas.data import DataSourceConfigSchema

        with pytest.raises(ValidationError, match="cannot be the set"):
            DataSourceConfigSchema(**{twin: "same.json", "test_index_path": "same.json"})

    def test_distinct_test_index_is_kept(self):
        from spectramr.config.schemas.data import DataSourceConfigSchema

        src = DataSourceConfigSchema(
            index_path="train.json", validation_index_path="val.json", test_index_path="test.json"
        )
        assert src.test_index_path == "test.json"


class TestRepPairTargetMode:
    """``target_mode: rep_pair`` is the leave-one-out pair by definition."""

    def test_rep_pair_requires_the_input_to_be_excluded(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        with pytest.raises(ValueError, match="rep_pair"):
            DataConfigSchema(target_mode="rep_pair")

    def test_rep_pair_with_exclusion_loads(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(target_mode="rep_pair", nex_target_exclude_input=True)
        assert cfg.target_mode == "rep_pair"


class TestSliceLevelRecords:
    """``data.slice_level_records`` (#1757) is declared, off by default, and
    read only by the m4raw route. ``DataConfigSchema`` is ``extra="ignore"``,
    so an undeclared spelling would be swallowed silently: the field's
    presence is itself the contract."""

    def test_is_a_declared_field_that_defaults_off(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        assert "slice_level_records" in DataConfigSchema.model_fields
        assert DataConfigSchema().slice_level_records is False

    def test_on_an_m4raw_arm_it_loads(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(dataset_type="m4raw", slice_level_records=True)
        assert cfg.slice_level_records is True

    def test_on_any_other_route_it_is_refused(self) -> None:
        """Planted violation: the knob would be unread (non-negotiable 8)."""
        from spectramr.config.schemas.data import DataConfigSchema

        with pytest.raises(ValueError, match="read only by dataset_type 'm4raw'"):
            DataConfigSchema(dataset_type="kspace", slice_level_records=True)

    def test_an_alias_of_m4raw_is_canonicalised_before_the_gate(self) -> None:
        """``mode="after"``: the alias has become ``m4raw`` by the time the
        route gate compares, so the spelling does not decide the verdict."""
        from spectramr.config.schemas.data import DATASET_TYPE_ALIASES, DataConfigSchema

        alias = next(k for k, v in DATASET_TYPE_ALIASES.items() if v == "m4raw")
        cfg = DataConfigSchema(dataset_type=alias, slice_level_records=True)
        assert cfg.dataset_type == "m4raw" and cfg.slice_level_records is True

    def test_off_on_any_route_is_fine(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        assert (
            DataConfigSchema(dataset_type="kspace", slice_level_records=False).slice_level_records
            is False
        )


class TestR2RTargetMode:
    """``target_mode: 'r2r'`` and the two nonlinearities that silently break it.

    R2R's guarantee is second-moment: ``Cov(y + a z, y - z/a) = 0``, which makes
    the L1/L2 minimiser the CLEAN image even though no clean image is ever shown.
    That identity survives LINEAR maps (``Cov(A a, A b) = A Cov(a, b) A^H``) and
    dies under a nonlinearity -- SILENTLY, because the loss still falls. So both
    nonlinear stages in this pipeline raise here rather than degrade (NN#3).
    """

    def test_r2r_is_in_the_target_mode_vocabulary(self):
        assert DataConfigSchema(target_mode="r2r", val_target_mode="phase_aligned_mean").target_mode == "r2r"

    def test_r2r_alpha_defaults_to_one(self):
        """alpha=1 splits the noise evenly: 2*Sigma into each half."""
        assert DataConfigSchema(target_mode="r2r", val_target_mode="phase_aligned_mean").r2r_alpha == 1.0

    def test_r2r_alpha_is_accepted_under_r2r(self):
        assert DataConfigSchema(target_mode="r2r", r2r_alpha=0.5, val_target_mode="phase_aligned_mean").r2r_alpha == 0.5

    @pytest.mark.parametrize("bad_alpha", [0.0, -1.0])
    def test_non_positive_alpha_is_refused(self, bad_alpha):
        """alpha appears as ``1/alpha`` in the target, so 0 is not a limit."""
        with pytest.raises(ValidationError):
            DataConfigSchema(target_mode="r2r", r2r_alpha=bad_alpha, val_target_mode="phase_aligned_mean")

    @pytest.mark.parametrize("other_mode", ["complex_mean", "phase_aligned_mean", "rep_pair"])
    def test_alpha_declared_under_another_mode_is_refused(self, other_mode):
        """An unread knob is a defect, not a harmless extra (NN#8)."""
        with pytest.raises(ValidationError, match="r2r_alpha"):
            DataConfigSchema(target_mode=other_mode, r2r_alpha=2.0)

    def test_alpha_declared_with_no_mode_at_all_is_refused(self):
        with pytest.raises(ValidationError, match="r2r_alpha"):
            DataConfigSchema(r2r_alpha=2.0)

    def test_r2r_without_a_validation_reference_is_refused(self):
        """An r2r validation target is a fresh random image every epoch.

        PSNR against it measures the noise draw, and ``track_best_metric`` would
        checkpoint whichever epoch drew kindest. The knob is not optional under
        r2r, so the schema names it rather than defaulting (NN#3).
        """
        with pytest.raises(ValidationError, match="val_target_mode"):
            DataConfigSchema(target_mode="r2r")

    def test_r2r_with_leave_one_out_is_accepted(self):
        """Pins a DELETED guard: this combination used to raise, and must not.

        ``nex_target_exclude_input`` is ONE knob shared by both splits. Under
        r2r the training split never averages -- but the VALIDATION split runs
        ``val_target_mode`` (a real averaging mode), and leave-one-out there is
        what keeps the input repetition out of its own reference. Refusing the
        pair made the whole cohort's arm A unconstructible.
        """
        cfg = DataConfigSchema(
            target_mode="r2r", val_target_mode="phase_aligned_mean",
            nex_target_exclude_input=True,
        )
        assert cfg.nex_target_exclude_input is True
        assert cfg.val_target_mode == "phase_aligned_mean"

    def test_val_target_mode_defaults_to_none(self):
        """None means "serve the training mode to both splits"."""
        assert DataConfigSchema().val_target_mode is None

    def test_val_target_mode_cannot_be_r2r(self):
        """Excluded at the TYPE level: a redrawn target is not a metric reference."""
        with pytest.raises(ValidationError):
            DataConfigSchema(target_mode="r2r", val_target_mode="r2r")

    @pytest.mark.parametrize(
        "train_mode", ["complex_mean", "phase_aligned_mean", "rep_pair"]
    )
    def test_val_target_mode_is_readable_under_every_training_mode(self, train_mode):
        """It is not an r2r-only knob.

        Holding ONE val reference fixed across a cohort is what makes the arms
        comparable, so every arm declares it -- not just the r2r one.
        """
        extra = {"nex_target_exclude_input": True} if train_mode == "rep_pair" else {}
        cfg = DataConfigSchema(
            target_mode=train_mode, val_target_mode="phase_aligned_mean", **extra
        )
        assert cfg.val_target_mode == "phase_aligned_mean"
        assert cfg.target_mode == train_mode

    @pytest.mark.parametrize("mode", ["rss", "magnitude", "svd"])
    def test_nonlinear_coil_combine_is_refused_under_r2r(self, mode):
        """``|.|`` maps the Gaussian residual to Rician: E[target] != x.

        This is the guard that matters most in practice -- ``dataset_type: m4raw``
        injects ``coils.processing_mode: svd`` through a preset, so an r2r arm
        that does not say ``none`` explicitly gets a nonlinear combine it never
        asked for.
        """
        with pytest.raises(ValidationError, match="processing_mode"):
            DataConfigSchema(target_mode="r2r", coils={"processing_mode": mode}, val_target_mode="phase_aligned_mean")

    def test_linear_coil_path_is_accepted_under_r2r(self):
        cfg = DataConfigSchema(target_mode="r2r", coils={"processing_mode": "none"}, val_target_mode="phase_aligned_mean")
        assert cfg.coils.processing_mode == "none"

    def test_log_scaling_is_refused_under_r2r(self):
        """``log1p`` on magnitude breaks decorrelation the same way a modulus does."""
        with pytest.raises(ValidationError, match="enable_log_scaling"):
            DataConfigSchema(
                target_mode="r2r",
                coils={"processing_mode": "none"},
                processing={"enable_log_scaling": True},
                val_target_mode="phase_aligned_mean",
            )

    @pytest.mark.parametrize("mode", ["complex_mean", "phase_aligned_mean", "rep_pair"])
    def test_the_linearity_guards_do_not_fire_for_other_modes(self, mode):
        """The guards are r2r-scoped: every other arm keeps its nonlinear stages.

        Without this, the change would be a corpus-wide break dressed as a new
        feature -- most M4Raw arms DO use a magnitude coil combine.
        """
        extra = {"nex_target_exclude_input": True} if mode == "rep_pair" else {}
        cfg = DataConfigSchema(
            target_mode=mode,
            coils={"processing_mode": "rss"},
            processing={"enable_log_scaling": True},
            **extra,
        )
        assert cfg.target_mode == mode
