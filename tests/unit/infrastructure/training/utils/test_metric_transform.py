"""Tests for the single metric-transform resolver (issue #931).

The resolver replaces three separate answers to "which transform grades this
run?" -- one in ``_apply_metric_transforms``, one in the ``metrics.transform``
schema field that nothing read, and one in
``check_metric_domain_matches_loss_output``, which granted a pass to any arm
declaring the inert knob.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.utils.metric_transform import (
    IMPLEMENTED_METRIC_TRANSFORMS,
    canonical_metric_transform,
    declared_metric_transforms,
    resolve_metric_transform,
)


def _validation(output_transform=None, domain=None, transform=None):
    """A validation config with the post-decomposition ``scoring`` sub-block."""
    return SimpleNamespace(
        scoring=SimpleNamespace(
            output_transform=output_transform,
            domain=domain,
            transform=transform,
        )
    )


def _metrics(transform=None, domain=None):
    return SimpleNamespace(transform=transform, domain=domain)


class TestCanonicalMetricTransform:
    """Name normalisation: aliases, sentinels, casing."""

    @pytest.mark.parametrize("spelling", ["ifft_mag_combine", "ifft_mag"])
    def test_the_dead_spellings_are_not_aliased_to_ifft_magnitude(self, spelling):
        """146 arms declared these, and aliasing them would have been a regression.

        Past tense since the #937 drain: 3 remain, held on #986. The reasoning
        below is why they were not aliased, and it still governs the next one.


        The temptation is real: the implementation deleted in e57c21021 was
        pair-R/I -> ifft2c -> abs -> RSS, operationally identical to today's
        ``ifft_magnitude``. But aliasing activates all 146, and 112 of them
        have ``losses.policy.output_domain`` and ``infer_output_domain`` BOTH
        equal to ``image`` — an IFFT there yields a Fourier magnitude, not a
        combine. (``losses.output_domain`` is the RETIRED leaf, folded to
        ``losses.policy.output_domain``; reading it post-fold gives ``None`` on
        every arm, which silently collapses the partition — see #982.)
        They must raise so a human decides per arm.
        """
        assert canonical_metric_transform(spelling) == spelling
        assert spelling not in IMPLEMENTED_METRIC_TRANSFORMS

    @pytest.mark.parametrize("sentinel", ["none", "NONE", "None", "", "  ", None])
    def test_none_sentinels_normalise_to_no_transform(self, sentinel):
        """``'none'`` is a truthy string.

        Before this change it survived as ``transform_name``, matched no
        dispatcher branch, and fell out of the bottom unchanged. Once an
        unknown name raises, that silent pass-through becomes a crash for the
        21 arms declaring it -- so the sentinel must normalise to ``None``.
        """
        assert canonical_metric_transform(sentinel) is None

    def test_implemented_name_passes_through(self):
        assert canonical_metric_transform("ifft_sense_adjoint") == "ifft_sense_adjoint"

    def test_casing_is_normalised(self):
        assert canonical_metric_transform("IFFT_Magnitude") == "ifft_magnitude"

    def test_unknown_name_is_returned_not_swallowed(self):
        """An unknown name must survive resolution so the caller can raise."""
        assert canonical_metric_transform("ifft_wavelet") == "ifft_wavelet"


class TestResolutionPrecedence:
    def test_validation_scoring_beats_metrics_transform(self):
        """The 67 arms that declare both, disagreeing.

        ``validation.scoring.output_transform`` is the canonical knob, so
        wiring ``metrics.transform`` must not change what they measure.
        """
        resolution = resolve_metric_transform(
            _validation(output_transform="ifft_magnitude"),
            _metrics(transform="ifft_sense_adjoint"),
        )
        assert resolution.name == "ifft_magnitude"
        assert resolution.source == "validation.scoring.output_transform"
        # and the losing declaration is still reported, for the audit
        assert ("metrics.transform", "ifft_sense_adjoint") in declared_metric_transforms(
            _validation(output_transform="ifft_magnitude"),
            _metrics(transform="ifft_sense_adjoint"),
        )

    def test_metrics_transform_never_supplies_a_name_on_this_path(self):
        """``metrics.transform`` reaches the dispatcher only via the TRAINING path.

        There the ``metrics`` block is passed as the first argument by
        ``_compute_training_metrics``. Letting the second argument also supply a
        name would activate it on the validation path too — 146 extra arms, 112
        of them onto an image output. It contributes suppression only.
        """
        resolution = resolve_metric_transform(
            _validation(), _metrics(transform="ifft_sense_adjoint")
        )
        assert resolution.name is None
        assert resolution.source is None

    def test_the_training_path_still_reads_metrics_transform(self):
        """Passed in the first slot, exactly as _compute_training_metrics does."""
        resolution = resolve_metric_transform(_metrics(transform="ifft_sense_adjoint"))
        assert resolution.name == "ifft_sense_adjoint"

    def test_the_two_stage_fallback_is_preserved(self):
        """The dispatcher always fell back to ``self.config.validation``.

        Callers hand in a config; only if it declares nothing is the strategy's
        own validation block consulted. Collapsing that to one source would
        silently drop the fallback for a narrowed or per-level config.
        """
        resolution = resolve_metric_transform(
            _validation(),
            None,
            fallback_validation_config=_validation(output_transform="ifft_magnitude"),
        )
        assert resolution.name == "ifft_magnitude"

    def test_caller_override_wins_over_config(self):
        """``diffusion.py`` resolves the name itself and hands it straight in."""
        resolution = resolve_metric_transform(
            _validation(output_transform="ifft_magnitude"),
            _metrics(transform="ifft_sense_adjoint"),
            caller_override="magnitude",
        )
        assert resolution.name == "magnitude"
        assert resolution.source == "caller"

    def test_explicit_none_at_higher_precedence_stops_the_cascade(self):
        """Declaring the winning key AT ALL is the decision.

        ``output_transform: none`` means "do not transform", so it must not
        fall through to ``metrics.transform`` and resurrect one the author
        turned off. No arm in the corpus pairs these today; the rule is fixed
        here so the first one that does behaves predictably.
        """
        resolution = resolve_metric_transform(
            _validation(output_transform="none"),
            fallback_validation_config=_validation(output_transform="ifft_magnitude"),
        )
        assert resolution.name is None
        assert resolution.source == "validation.scoring.output_transform"

    def test_absent_is_distinct_from_declared_none(self):
        """An unset field is ``None``; a declared sentinel is the str 'none'."""
        resolution = resolve_metric_transform(
            _validation(output_transform=None),
            fallback_validation_config=_validation(output_transform="ifft_magnitude"),
        )
        assert resolution.name == "ifft_magnitude"

    def test_nothing_declared_resolves_to_nothing(self):
        resolution = resolve_metric_transform(_validation(), _metrics())
        assert resolution.name is None
        assert resolution.source is None
        assert not resolution.suppressed


class TestSuppression:
    def test_domain_none_suppresses_everything(self):
        """``domain: none`` is the explicit off switch.

        It outranks a declared transform and also bypasses the auto-magnitude
        gate, which is why it is reported separately from ``name is None``.
        """
        resolution = resolve_metric_transform(
            _validation(domain="none", output_transform="ifft_magnitude"), _metrics()
        )
        assert resolution.suppressed

    def test_metrics_domain_none_also_suppresses(self):
        resolution = resolve_metric_transform(
            _validation(), _metrics(transform="ifft_magnitude", domain="none")
        )
        assert resolution.suppressed

    def test_ordinary_domain_does_not_suppress(self):
        resolution = resolve_metric_transform(
            _validation(domain="image", output_transform="ifft_magnitude"), _metrics()
        )
        assert not resolution.suppressed
        assert resolution.name == "ifft_magnitude"


class TestImplementedPredicate:
    def test_known_name_is_implemented(self):
        assert resolve_metric_transform(
            _validation(output_transform="ifft_sense_adjoint"), None
        ).is_implemented

    def test_unknown_name_is_not_implemented(self):
        """``fft`` is advertised by two schema descriptions and dispatched by none."""
        resolution = resolve_metric_transform(_validation(output_transform="fft"), None)
        assert resolution.name == "fft"
        assert not resolution.is_implemented

    def test_no_transform_counts_as_implemented(self):
        """``None`` is not a name the dispatcher has to know."""
        assert resolve_metric_transform(_validation(), None).is_implemented


class TestConfigShapes:
    def test_flat_legacy_spelling_still_resolves(self):
        """``validation.output_transform`` folds into ``scoring`` at load.

        A Python read of the flat spelling returns None on a real config, but
        a hand-built stub may still use it -- and #927 was exactly this class
        of miss, so the resolver reads both.
        """
        flat = SimpleNamespace(output_transform="ifft_magnitude", scoring=None)
        assert resolve_metric_transform(flat, None).name == "ifft_magnitude"

    def test_mapping_configs_resolve(self):
        resolution = resolve_metric_transform(
            {"scoring": {"output_transform": "ifft_magnitude"}},
            {"transform": "ifft_sense_adjoint"},
        )
        assert resolution.name == "ifft_magnitude"

    def test_none_configs_are_safe(self):
        assert resolve_metric_transform(None, None).name is None


class TestCorpusResolves:
    """The acceptance gate: unit tests cannot tell you 236 arms still work.

    Raising on an unknown transform name (pitfall #9) turns any spelling the
    dispatcher does not implement from a silent no-op into a crash. The set of
    spellings actually in the corpus is therefore the thing under test, and it
    lives in the YAML, not in this file.
    """

    #: Arms declaring a transform no branch dispatches. A RATCHET: this may
    #: shrink as the drain lands, never grow. Do not "fix" a rise by editing
    #: the number.
    #:
    #: 146 when measured 2026-08-09 (143 ``ifft_mag_combine`` + 3 ``ifft_mag``);
    #: **3** now that the #937 drain has landed in full -- 112 image-domain arms
    #: had the key deleted (#983/#984/#985) and 31 k-space arms were pointed at
    #: ``ifft_magnitude`` on both metric paths (#987).
    #:
    #: The 3 that remain are held, not missed: ``imjense_pisco_siren_m4raw``,
    #: ``exp_vf_ib_infonce_v2`` and ``experiment_vf_tto_v2`` each resolve to
    #: ``image`` ONLY because their ``model_type`` sits in a hardcoded legacy set
    #: that ``infer_output_domain`` consults at P3 -- above P4's explicit
    #: ``data.domain.output``. All three declare ``dataset_type: kspace``.
    #: Draining them would bake that inversion in; see #986. **When #986 lands,
    #: this number moves again.**
    #:
    #: Tighten with every drain, or the ratchet stops having teeth: the assertion
    #: is ``<=``, so a successful drain leaves it loose rather than failing.
    #:
    #: Count by LOADING the document, never by matching the token --
    #: ``experiment_30_mamba_mri_reconstruction.yaml`` mentions
    #: ``ifft_mag_combine`` in a comment and declares no transform at all, so
    #: ``git grep -l`` overcounts the corpus by one (#982).
    UNDISPATCHABLE_BASELINE = 3

    #: The only two dead spellings. A NEW one appearing means someone invented a
    #: transform name against a schema description that no longer offers any.
    DEAD_SPELLINGS: frozenset[str] = frozenset({"ifft_mag_combine", "ifft_mag"})

    @staticmethod
    def _declared_transforms() -> list[tuple[str, str, str]]:
        """(arm, key, raw value) for every transform declared under inprogress/."""
        import yaml

        from tests.utils.corpus import repo_root, tracked_yamls

        found: list[tuple[str, str, str]] = []
        root = repo_root()
        for path in tracked_yamls(root / "experiments" / "inprogress"):
            try:
                doc = yaml.safe_load(path.read_text())
            except yaml.YAMLError:
                continue  # malformed YAML is a different ratchet's problem
            if not isinstance(doc, dict):
                continue
            arm = str(path.relative_to(root))
            metrics = doc.get("metrics")
            if isinstance(metrics, dict) and metrics.get("transform") is not None:
                found.append((arm, "metrics.transform", metrics["transform"]))
            validation = doc.get("validation")
            if not isinstance(validation, dict):
                continue
            scoring = validation.get("scoring")
            block = scoring if isinstance(scoring, dict) else validation
            for key in ("transform", "output_transform"):
                if block.get(key) is not None:
                    found.append((arm, f"validation.{key}", block[key]))
        return found

    def _undispatchable(self) -> list[tuple[str, str, str]]:
        return [
            (arm, key, raw)
            for arm, key, raw in self._declared_transforms()
            if (name := canonical_metric_transform(raw)) is not None
            and name not in IMPLEMENTED_METRIC_TRANSFORMS
        ]

    def test_the_corpus_declares_transforms_at_all(self):
        """Guard against a sweep that passes because it found nothing."""
        assert len(self._declared_transforms()) > 100

    def test_the_validation_path_declares_nothing_undispatchable(self):
        """#930 made ``output_transform`` fire; nothing there may now raise.

        Corpus-wide it is only ever ``ifft_magnitude`` (84) or ``none`` (2), so
        the raise cannot break a single validation-path arm. This is the check
        that says so rather than assuming it.
        """
        offenders = [
            row for row in self._undispatchable() if row[1].startswith("validation.")
        ]
        assert not offenders, offenders[:5]

    def test_undispatchable_declarations_only_shrink(self):
        """The 146-arm drain, pinned as a ratchet (3 remain, all held on #986).

        These arms name a transform that does not exist. Before this change the
        dispatcher swallowed it; now it raises and the audit fails them
        pre-flight. Each needs a human decision -- delete the key (the 112 with
        image-domain output) or name ``ifft_magnitude`` (the ~31 with k-space
        output) -- which is why they are tracked, not auto-migrated.
        """
        found = self._undispatchable()
        assert len(found) <= self.UNDISPATCHABLE_BASELINE, (
            f"{len(found)} undispatchable declarations, baseline "
            f"{self.UNDISPATCHABLE_BASELINE}. A new one was added: "
            f"{found[: self.UNDISPATCHABLE_BASELINE + 3][-3:]}"
        )

    def test_no_new_dead_spelling_was_invented(self):
        """Every undispatchable value is one of the two known dead spellings."""
        novel = {
            canonical_metric_transform(raw) for _, _, raw in self._undispatchable()
        } - self.DEAD_SPELLINGS
        assert not novel, f"unknown transform spellings appeared: {sorted(novel)}"


class TestTheAsymmetricTargetTransform:
    """``ifft_target_magnitude``: an image prediction against a k-space target (#2325)."""

    @staticmethod
    def _blob_kspace(coils: int = 2, size: int = 32) -> tuple[torch.Tensor, tuple[int, int]]:
        from spectramr.infrastructure.physics.fft_ops import fft2c

        image = torch.zeros(1, coils, size, size, dtype=torch.complex64)
        peak = (size // 4, 3 * size // 4)
        image[:, :, peak[0], peak[1]] = 1.0 + 0.5j
        kspace = fft2c(image)
        interleaved = torch.empty(1, 2 * coils, size, size)
        interleaved[:, 0::2], interleaved[:, 1::2] = kspace.real, kspace.imag
        return interleaved, peak

    def test_the_target_returns_to_the_image_where_its_peak_is(self):
        from spectramr.infrastructure.training.utils.metric_transform import kspace_rss_magnitude

        kspace, peak = self._blob_kspace()
        image = kspace_rss_magnitude(kspace)
        assert image.shape == (1, 1, 32, 32)
        flat = int(image.reshape(-1).argmax())
        assert (flat // 32, flat % 32) == peak

    def test_the_prediction_keeps_its_own_domain(self):
        from spectramr.infrastructure.training.utils.metric_transform import image_magnitude

        pred = torch.tensor([[[[3.0]], [[4.0]]]])
        assert torch.allclose(image_magnitude(pred), torch.tensor([[[[5.0]]]]))

    def test_an_uninterleavable_target_raises(self):
        from spectramr.infrastructure.training.utils.metric_transform import kspace_rss_magnitude

        with pytest.raises(ValueError, match="interleaved"):
            kspace_rss_magnitude(torch.randn(1, 3, 8, 8))

    def test_it_is_dispatchable(self):
        assert "ifft_target_magnitude" in IMPLEMENTED_METRIC_TRANSFORMS


def test_sense_magnitude_is_dispatchable() -> None:
    """Its dispatcher branch is exercised in ``test_metrics_mixin``; a name in the
    set with no branch would pass ``is_implemented`` and return the tensors raw."""
    assert "sense_magnitude" in IMPLEMENTED_METRIC_TRANSFORMS
    resolution = resolve_metric_transform(caller_override="SENSE_magnitude")
    assert resolution.name == "sense_magnitude" and resolution.is_implemented
