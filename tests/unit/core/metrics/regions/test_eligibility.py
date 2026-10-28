"""The eligibility gate: coverage (T1), the context invariant (T3), and no defaults.

The gate is the anti-facade core. Without it, a metric that is *undefined* on a
40x40 lesion ROI still returns a number, and the per-segment leaderboard becomes
noise dressed as science.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.outcome import NotApplicableReason  # noqa: E402
from spectramr.core.metrics.regions.eligibility import (  # noqa: E402
    ROI_POLICY,
    Normalisation,
    RoiPolicy,
    RoiSupport,
    evaluate_eligibility,
    policy_for,
)
from spectramr.core.metrics.regions.types import RegionMask, RegionSource  # noqa: E402
from spectramr.core.metrics.registry import MetricsRegistry  # noqa: E402


def _rect_region(h: int, w: int, region_id: str = "path:lesion#0") -> RegionMask:
    mask = torch.zeros(320, 320, dtype=torch.bool)
    mask[10 : 10 + h, 10 : 10 + w] = True
    return RegionMask(
        mask=mask,
        region_id=region_id,
        source=RegionSource.PATHOLOGY,
        provenance={"annotation": "test"},
    )


def _blob_region(region_id: str = "synthseg:gm") -> RegionMask:
    """A large NON-rectangular region (a disc) -- the GM/WM case."""
    yy, xx = torch.meshgrid(torch.arange(320), torch.arange(320), indexing="ij")
    mask = ((yy - 160) ** 2 + (xx - 160) ** 2) < 100**2
    return RegionMask(
        mask=mask,
        region_id=region_id,
        source=RegionSource.TISSUE,
        provenance={"segmenter": "synthseg"},
    )


class TestT1Coverage:
    """Every swept metric has a declared policy. This is what makes 'no default' safe."""

    def test_every_metric_spec_key_has_a_policy(self) -> None:
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import METRIC_SPECS

        missing = sorted(
            s.registry_key for s in METRIC_SPECS if s.registry_key not in ROI_POLICY
        )
        assert not missing, (
            f"{len(missing)} swept metric(s) have no declared ROI policy: {missing}. "
            "There is deliberately no default -- declare each one in "
            "core/metrics/regions/eligibility.py."
        )

    def test_no_hand_declared_group_names_a_metric_that_is_not_swept(self) -> None:
        """Guards the reverse drift: a stale hand-written entry.

        Only the hand-declared groups are checked. The context-metric policies are
        auto-derived from the whole registry (which is larger than METRIC_SPECS), so
        covering an unswept context metric there is correct, not stale.
        """
        from spectramr.core.metrics.regions import eligibility as elig
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import METRIC_SPECS

        swept = {s.registry_key for s in METRIC_SPECS}
        hand_declared = (
            {k for _, keys in elig._NOT_RESTRICTABLE_GROUPS.values() for k in keys}
            | set(elig._MAP_TIER)
            | set(elig._PERCEPTUAL)
            | set(elig._NO_REFERENCE)
            | set(elig._CROP_SCALAR)
        )
        stale = sorted(hand_declared - swept)
        assert (
            not stale
        ), f"hand-declared ROI policies for metrics that are not swept: {stale}"

    def test_every_not_restrictable_policy_carries_a_justification(self) -> None:
        """An ineligible metric without a written reason is an unexplained exclusion."""
        for key, policy in ROI_POLICY.items():
            if policy.support is RoiSupport.NOT_RESTRICTABLE:
                assert len(policy.justification) > 20, key


class TestPerImageSweptMetrics:
    """The two swept per-image metrics are the ones a regional sweep actually reaches."""

    def test_zipper_detection_crops_as_a_no_reference_statistic(self) -> None:
        """Reads the prediction alone, so it crops like the rest of the NR family."""
        policy = policy_for("zipper_detection")
        assert policy.support is RoiSupport.CROP_THEN_COMPUTE
        assert policy.min_roi_px > 0
        assert not evaluate_eligibility("zipper_detection", _rect_region(20, 20)).eligible

    def test_task_based_detectability_is_declined_on_a_region(self) -> None:
        """One 32 px tile fits a ~40 px lesion box; the CHO needs two to estimate a
        covariance, so a crop would return NaN or a ridge-determined d'."""
        v = evaluate_eligibility("task_based_detectability", _rect_region(41, 41))
        assert not v.eligible
        assert v.support is RoiSupport.NOT_RESTRICTABLE


class TestT3ContextInvariant:
    """The strongest rule: a k-space measurement cannot be restricted by an image crop."""

    def test_every_context_metric_is_not_restrictable(self) -> None:
        offenders = [
            key
            for key, policy in ROI_POLICY.items()
            if MetricsRegistry.needs_context(key)
            and policy.support is not RoiSupport.NOT_RESTRICTABLE
        ]
        assert not offenders, (
            f"{offenders} need a MetricContext (k-space / coil maps / sampling mask) "
            "but are marked ROI-restrictable. An image-space crop is not a k-space "
            "restriction -- that is a category error, not an approximation."
        )

    def test_a_physics_metric_is_declined_with_the_right_reason(self) -> None:
        v = evaluate_eligibility("g_factor", _blob_region())
        assert not v.eligible
        assert v.reason is NotApplicableReason.MEASUREMENT_NOT_ROI_RESTRICTABLE
        assert v.support is RoiSupport.NOT_RESTRICTABLE


class TestNoSafeDefault:
    def test_an_undeclared_metric_raises_rather_than_assuming_a_tier(self) -> None:
        """Assuming 'crop' would silently score new metrics on 40px crops; assuming
        'none' would make them silently vanish. Both fabricate. So: raise."""
        with pytest.raises(KeyError, match="no declared ROI policy"):
            policy_for("a_brand_new_metric")

    def test_a_policy_without_a_justification_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="human justification"):
            RoiPolicy(support=RoiSupport.CROP_THEN_COMPUTE, justification="")

    def test_a_not_restrictable_policy_cannot_carry_a_size_floor(self) -> None:
        """Size is not why it is ineligible -- conflating the two hides the real reason."""
        with pytest.raises(ValueError, match="cannot carry a size floor"):
            RoiPolicy(
                support=RoiSupport.NOT_RESTRICTABLE,
                justification="needs the measurement",
                min_roi_side=64,
            )


class TestSizeFloors:
    def test_ms_ssim_is_declined_on_a_typical_lesion_bbox(self) -> None:
        """T6. ~40x40 is the fastMRI+ bbox scale; MS-SSIM needs 161 px/side."""
        v = evaluate_eligibility("ms_ssim", _rect_region(41, 41))
        assert not v.eligible
        assert v.reason is NotApplicableReason.ROI_TOO_SMALL_SIDE
        assert "161" in v.detail

    def test_ms_ssim_is_fine_on_the_full_slice(self) -> None:
        full = RegionMask(
            mask=torch.ones(320, 320, dtype=torch.bool),
            region_id="full",
            source=RegionSource.FULL,
        )
        assert evaluate_eligibility("ms_ssim", full).eligible

    def test_a_patch_statistic_is_declined_on_too_small_an_area(self) -> None:
        """min_side and min_px are both needed: a 1x3000 sliver passes an area floor
        while having no usable neighbourhood, and a small square passes a side floor
        while having no usable sample."""
        # brisque has an AREA floor (1024 px) and no side floor, so a 20x20 = 400 px
        # square is declined for area specifically. (niqe now also carries a 96 px SIDE
        # floor -- its patch grid is 96x96 -- so it trips the side rule first.)
        v = evaluate_eligibility("brisque", _rect_region(20, 20))
        assert not v.eligible
        assert v.reason is NotApplicableReason.ROI_TOO_SMALL_AREA

    def test_map_tier_metrics_have_no_size_floor(self) -> None:
        """MSE on 40x40 is exactly MSE on 40x40. No floor is needed and none is imposed."""
        v = evaluate_eligibility("mse", _rect_region(41, 41))
        assert v.eligible
        assert v.support is RoiSupport.MAP_THEN_MASK


class TestNonRectangularHazard:
    def test_a_crop_tier_metric_is_declined_on_a_non_rectangular_region(self) -> None:
        """The bbox of GM is a strict SUPERSET of GM.

        Cropping to it would score the bbox -- which contains WM, CSF and background
        -- and report it as "LPIPS on grey matter". Better to decline than to answer
        a different question.
        """
        v = evaluate_eligibility("lpips", _blob_region())
        assert not v.eligible
        assert v.reason is NotApplicableReason.ROI_NOT_RECTANGULAR

    def test_a_map_tier_metric_is_fine_on_a_non_rectangular_region(self) -> None:
        """Map-then-mask has no such problem: it reduces over the mask itself."""
        v = evaluate_eligibility("ssim", _blob_region())
        assert v.eligible
        assert v.support is RoiSupport.MAP_THEN_MASK


class TestNormalisationIsStamped:
    def test_map_tier_declares_full_slice_normalisation(self) -> None:
        """The string lands on every output row, so a reader can check comparability."""
        v = evaluate_eligibility("psnr", _blob_region())
        assert v.normalisation is Normalisation.FULL_SLICE
        assert v.as_row()["normalisation"] == "full_slice"


class TestExclusionTableIsPure:
    def test_a_verdict_needs_no_tensors_and_calls_no_metric(self) -> None:
        """So a run can emit its complete exclusion table BEFORE computing anything."""
        row = evaluate_eligibility("ms_ssim", _rect_region(41, 41)).as_row()
        assert row["metric"] == "ms_ssim"
        assert row["eligible"] is False
        assert row["min_side"] == 41
        assert row["tier"] == "crop"
        assert row["reason"] == "roi_min_side_below_metric_floor"


class TestFloorsComeFromTheMetric:
    """A declared floor that disagrees with the metric's implementation is not a gate.

    `iw_ssim` declares `_MIN_SIZE = 161` on its own class and `_PIQMetric._maybe_upsample`
    BILINEARLY INTERPOLATES anything smaller up to it. The gate's hand-kept table listed
    only `ms_ssim`, so `iw_ssim` on a 40x40 lesion crop returned a confident 0.9323 --
    computed on invented high-frequency detail. That is the exact fabrication the package
    exists to decline, and a fastMRI+ lesion box is ~40 px, so it was every lesion cell.
    """

    def test_a_metric_that_would_upsample_is_declined_not_upsampled(self) -> None:
        for key in ("ms_ssim", "iw_ssim"):
            v = evaluate_eligibility(key, _rect_region(40, 40))
            assert not v.eligible, (
                f"{key} accepted a 40x40 ROI, but its backend upsamples to 161x161 and "
                "returns a confident score on interpolated content."
            )
            assert v.reason is NotApplicableReason.ROI_TOO_SMALL_SIDE

    def test_every_declared_min_size_is_honoured_by_the_gate(self) -> None:
        """The floor is read OFF the metric class, so the two cannot drift apart."""
        from spectramr.core.metrics.regions.eligibility import ROI_POLICY, RoiSupport
        from spectramr.core.metrics.registry import MetricsRegistry

        for key, policy in ROI_POLICY.items():
            if policy.support is not RoiSupport.CROP_THEN_COMPUTE:
                continue
            cls = MetricsRegistry._metrics.get(key)
            declared = int(getattr(cls, "_MIN_SIZE", 0) or 0) if cls is not None else 0
            if declared:
                assert policy.min_roi_side >= declared, (
                    f"{key} declares _MIN_SIZE={declared} and silently upsamples below "
                    f"it, but the gate's floor is {policy.min_roi_side}."
                )


class TestThePolicyTableIsNotFrozenAtImport:
    """The table is built lazily, so it cannot freeze against a half-filled registry.

    It used to be a module constant (``ROI_POLICY = _build_policy_table()``) evaluated at
    import -- and ``core/metrics/__init__`` imports this package partway through its own
    registration pass, so the table froze at 137 of 201 metrics. Every floor read off a
    metric class came out 0 (``iw_ssim`` then ACCEPTED a 40x40 crop and its backend
    upsampled it into a fabricated score) and the 27 context metrics were missing outright
    (``policy_for`` raised KeyError mid-sweep). Both were pure import-order artefacts,
    which is why the suite passed in one invocation and failed in another.
    """

    def test_the_late_registering_context_metrics_are_in_the_table(self) -> None:
        from spectramr.core.metrics.regions.eligibility import ROI_POLICY

        # These four register AFTER this package is imported, so a table frozen at import
        # time did not contain them at all. Not "declined" -- absent, which made
        # policy_for() raise KeyError partway through a sweep.
        for key in ("ndcr", "are", "csoe", "prp_snr"):
            assert key in ROI_POLICY, (
                f"{key} registers late; a table frozen at import time misses it and "
                "policy_for() then raises KeyError in the middle of a sweep."
            )
            assert policy_for(key).support is RoiSupport.NOT_RESTRICTABLE

    def test_a_class_declared_floor_survives_the_build(self) -> None:
        """The regression that matters: floor 0 means the gate lets a fabricator through."""
        from spectramr.core.metrics.regions.eligibility import ROI_POLICY, min_side_for

        assert min_side_for("iw_ssim") == 161
        assert ROI_POLICY["iw_ssim"].min_roi_side == 161, (
            "iw_ssim's floor came back 0 -- the table was built before its class was "
            "registered. A 0 floor makes the gate ACCEPT a 40x40 lesion crop, which the "
            "piq backend then bilinearly upsamples to 161x161 and scores confidently."
        )

    def test_the_table_rebuilds_when_the_registry_grows(self) -> None:
        import spectramr.core.metrics.regions.eligibility as elig

        elig._POLICY_TABLE = {}  # simulate a table built against an empty registry
        elig._BUILT_AT_REGISTRY_SIZE = 0
        assert elig.policy_for("ssim").support is RoiSupport.MAP_THEN_MASK


class TestTheFullRegionGetsNoExemption:
    """`full` is the control cell every regional ranking is compared against.

    It used to short-circuit to eligible=True before the size floors, so on a 128px slice
    it handed ms_ssim and iw_ssim an image below their 161px floor -- both upsample and
    fabricate. A floor is a property of the METRIC, not of the region.
    """

    def _full(self, side: int) -> RegionMask:
        return RegionMask(
            mask=torch.ones(side, side, dtype=torch.bool),
            region_id="full",
            source=RegionSource.FULL,
        )

    def test_a_small_full_slice_is_declined_not_upsampled(self) -> None:
        v = evaluate_eligibility("ms_ssim", self._full(128))
        assert not v.eligible
        assert v.reason is NotApplicableReason.ROI_TOO_SMALL_SIDE

    def test_a_full_size_slice_still_passes(self) -> None:
        assert evaluate_eligibility("ms_ssim", self._full(320)).eligible


class TestTemporalSeriesGroup:
    """tsnr / temporal_fidelity are NOT_RESTRICTABLE, and the reason is specific.

    Declaring them in METRIC_SPECS (so the health gate stops feeding them an image
    pair) also opts them into this table — there is deliberately no default. The
    honest tier is NOT_RESTRICTABLE, but *not* because they are parametric maps
    (they are not; they are computed from magnitude series). The reason is that a
    spatial ROI restricts to a single frame, which has no temporal axis at all.
    """

    def test_both_are_declared_not_restrictable(self) -> None:
        from spectramr.core.metrics.regions.eligibility import ROI_POLICY, RoiSupport

        for key in ("tsnr", "temporal_fidelity"):
            assert key in ROI_POLICY, f"{key} lost its ROI policy"
            assert ROI_POLICY[key].support is RoiSupport.NOT_RESTRICTABLE

    def test_they_are_not_filed_under_parametric_map(self) -> None:
        """Guard against the convenient-but-false justification.

        parametric_map says "defined on a parametric / non-magnitude map". These
        two are defined on a magnitude time series, so that group would be a
        wrong argument that happens to yield the right tier — and the next reader
        would believe it.
        """
        from spectramr.core.metrics.regions import eligibility as elig

        _, keys = elig._NOT_RESTRICTABLE_GROUPS["parametric_map"]
        assert "tsnr" not in keys and "temporal_fidelity" not in keys

        why_temporal, temporal_keys = elig._NOT_RESTRICTABLE_GROUPS["temporal_series"]
        assert set(temporal_keys) == {"tsnr", "temporal_fidelity"}
        assert "time axis" in why_temporal


class TestSecondPassDeclarations:
    """The seven metrics declared after the #368 second-pass ruling.

    Two are placed by analogy to an existing sibling; five are NOT_RESTRICTABLE per
    an explicit maintainer ruling (segmentation-overlap family + focal_frequency).
    Pinned here so a later edit cannot silently re-tier them or drop a declaration.
    """

    def test_lpips_alex_crops_like_lpips_with_the_deep_net_floor(self) -> None:
        # LPIPS with an AlexNet backbone: a deep-network perceptual metric, so it
        # crops the tight bbox and carries the 64 px deep-net floor like lpips/dists.
        p = policy_for("lpips_alex")
        assert p.support is RoiSupport.CROP_THEN_COMPUTE
        assert p.min_roi_side >= 64

    def test_nrmse_l2_crops_like_nrmse_not_maps(self) -> None:
        # nrmse_l2 shares nrmse's global ||target|| denominator, which has no verified
        # (maps, reduce) pair in reductions.py -- so it crops exactly like its sibling
        # nrmse and is deliberately NOT in the map tier.
        assert policy_for("nrmse_l2").support is RoiSupport.CROP_THEN_COMPUTE
        assert policy_for("nrmse").support is RoiSupport.CROP_THEN_COMPUTE
        assert "nrmse_l2" not in _map_tier_keys()

    @pytest.mark.parametrize(
        "key",
        [
            "focal_frequency",
            "brain_mask_dice",
            "tissue_dice",
            "tissue_hd95",
            "tissue_volume_similarity",
        ],
    )
    def test_ruled_not_restrictable(self, key: str) -> None:
        # focal_frequency: spectral (an image-space crop's FFT is not a k-space
        # restriction). The four seg-overlap metrics consume label maps; a lesion
        # bbox severs the structure and its boundary, so a restricted score answers
        # a different question rather than approximating the same one.
        policy = policy_for(key)
        assert policy.support is RoiSupport.NOT_RESTRICTABLE
        assert len(policy.justification) > 20
        v = evaluate_eligibility(key, _blob_region())
        assert not v.eligible
        assert v.reason is NotApplicableReason.MEASUREMENT_NOT_ROI_RESTRICTABLE


def _map_tier_keys() -> set[str]:
    from spectramr.core.metrics.regions import eligibility as elig

    return set(elig._MAP_TIER)
