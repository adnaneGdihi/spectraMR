"""``build_snapshot_provenance``: declared knobs beside what was actually applied.

The record exists so a reader can falsify a snapshot against the config — an
``input_prepared`` that looks fully sampled is a defect only if the arm declared
acceleration. These tests pin the two properties that make it trustworthy: it
never lies by omission (gaps land in ``incomplete``), and it never leaks an
unresolved object into the artifact.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.snapshot_provenance import (  # noqa: E402
    build_snapshot_provenance,
)


def _config(**processing_overrides):
    """A config shaped like the real SSOT: ``data.processing`` is a sub-block."""
    processing = SimpleNamespace(
        enable_kspace_normalization=True,
        kspace_percentile=0.99,
        enable_log_scaling=False,
        normalization_type="percentile",
        normalization_kwargs={"percentile": 0.99},
    )
    for key, value in processing_overrides.items():
        setattr(processing, key, value)
    return SimpleNamespace(
        data=SimpleNamespace(
            target_mode="phase_aligned_mean",
            val_target_mode=None,
            r2r_alpha=1.0,
            nex_target_exclude_input=False,
            nex_fallback="error",
            slice_level_records=False,
            processing=processing,
            augmentation=SimpleNamespace(enabled=True),
        )
    )


class _FakeTransform:
    """Duck-types a TorchIO leaf transform: a name plus ``args_names``."""

    args_names = ("p", "axes")

    def __init__(self, p: float = 0.5, axes: tuple[int, ...] = (0, 1)) -> None:
        self.p = p
        self.axes = axes


class _FakeCompose:
    """Duck-types ``tio.Compose``: a container exposing ``.transforms``."""

    def __init__(self, transforms) -> None:
        self.transforms = transforms


class _FakeDataset:
    def __init__(self, transform=None) -> None:
        self.transform = transform

    def __getitem__(self, idx):  # pragma: no cover - protocol only
        return idx


class _FakeWrapper:
    """Duck-types ``SFCConformalFMRIKeysWrapper``: delegates through ``inner``."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def __getitem__(self, idx):  # pragma: no cover - protocol only
        return self.inner[idx]


@pytest.mark.unit
class TestDeclaredHalf:
    def test_it_records_the_knobs_that_shape_input_and_target(self) -> None:
        record = build_snapshot_provenance(_config())
        declared = record["declared"]
        # `target_mode` is the knob behind the original report: a
        # phase_aligned_mean target is a coherent NEX average while the input is
        # one noisier repetition, so the two SHOULD look different.
        assert declared["target_mode"] == "phase_aligned_mean"
        assert declared["normalization"]["enable_kspace_normalization"] is True
        assert declared["normalization"]["normalization_kwargs"] == {"percentile": 0.99}
        assert declared["augmentation_enabled"] is True

    def test_a_missing_processing_block_is_reported_not_silently_empty(self) -> None:
        config = SimpleNamespace(data=SimpleNamespace(target_mode="complex_mean"))
        record = build_snapshot_provenance(config)
        assert "normalization" not in record["declared"]
        assert any("processing" in gap for gap in record["incomplete"])

    def test_a_knob_json_cannot_carry_is_named_not_quietly_dropped(self) -> None:
        """A declared-but-unrepresentable knob must not read as an unset one.

        `_scalarize` returns `None` both for "genuinely null" and for "I cannot
        coerce this". Dropping both left a record that looked like a complete
        account of the declaration while a knob that shaped the data went
        unmentioned — the partial-provenance facade of pitfall #16, produced by
        the very module written to prevent it.
        """

        class _Opaque:
            pass

        record = build_snapshot_provenance(_config(normalization_kwargs=_Opaque()))
        declared = record["declared"]["normalization"]
        assert "normalization_kwargs" not in declared
        gaps = record["incomplete"]
        assert any("normalization_kwargs" in gap and "_Opaque" in gap for gap in gaps), gaps

    def test_a_genuinely_unset_knob_is_not_reported_as_a_gap(self) -> None:
        """An unset knob is honestly absent. Listing every `None` would bury
        the real gaps in noise, which costs the `incomplete` list its meaning."""
        record = build_snapshot_provenance(_config(normalization_kwargs=None))
        assert "normalization_kwargs" not in record["declared"]["normalization"]
        assert not any("normalization_kwargs" in gap for gap in record["incomplete"])


@pytest.mark.unit
class TestAppliedHalf:
    def test_a_nested_compose_flattens_in_order_with_its_parameters(self) -> None:
        inner = _FakeCompose([_FakeTransform(p=0.25)])
        dataset = _FakeDataset(_FakeCompose([_FakeTransform(p=0.9), inner]))
        record = build_snapshot_provenance(_config(), dataset=dataset)
        transforms = record["applied"]["transforms"]
        assert [t["name"] for t in transforms] == ["_FakeTransform", "_FakeTransform"]
        assert transforms[0]["params"]["p"] == 0.9
        assert transforms[1]["params"]["p"] == 0.25
        assert record["incomplete"] == []

    def test_a_wrapper_is_named_and_the_walk_reaches_the_inner_transform(self) -> None:
        """`SFCConformalFMRIKeysWrapper` attaches keys OUTSIDE `.transform`.

        A record built from the Compose alone would omit it, which is the
        partial-provenance facade this module exists to avoid.
        """
        dataset = _FakeWrapper(_FakeDataset(_FakeCompose([_FakeTransform()])))
        record = build_snapshot_provenance(_config(), dataset=dataset)
        assert record["applied"]["dataset_chain"] == ["_FakeWrapper", "_FakeDataset"]
        assert len(record["applied"]["transforms"]) == 1

    def test_no_dataset_is_an_admission_not_an_empty_list(self) -> None:
        record = build_snapshot_provenance(_config(), dataset=None)
        assert record["applied"] == {}
        assert any("dataset" in gap for gap in record["incomplete"])

    def test_a_dataset_without_a_transform_says_so(self) -> None:
        record = build_snapshot_provenance(_config(), dataset=_FakeDataset(None))
        assert "transforms" not in record["applied"]
        assert any(".transform" in gap for gap in record["incomplete"])


class _FakeQueue:
    """Duck-types ``tio.Queue``: delegates through ``subjects_dataset`` ONLY.

    The real ``Queue`` carries no ``inner``/``dataset``/``base_dataset`` at all
    (verified on torchio 1.2.1 — all three are ``None``), so a walk that knows
    only those names stops dead at the ``Queue``.
    """

    def __init__(self, subjects_dataset) -> None:
        self.subjects_dataset = subjects_dataset

    def __getitem__(self, idx):  # pragma: no cover - protocol only
        return idx


class _FakeCallableCompose(_FakeCompose):
    """``tio.Compose`` is callable; the private read is gated on that."""

    def __call__(self, subject):  # pragma: no cover - protocol only
        return subject


class _FakeTorchIODataset:
    """Duck-types torchio >= 1.2 ``SubjectsDataset``: the transform is PRIVATE.

    1.2 privatized it — ``self._transform`` plus ``set_transform()``, with no
    public read accessor and no property. Reading only ``.transform`` therefore
    reports "no transform" for every one of the nine ``tio.SubjectsDataset``
    call sites in this repo.
    """

    def __init__(self, transform=None) -> None:
        self._transform = transform

    def __getitem__(self, idx):  # pragma: no cover - protocol only
        return idx


@pytest.mark.unit
class TestTheAppliedHalfSeesThroughTheQueue:
    """The coverage hole that made W2's declared-vs-applied divergence invisible.

    Every patch-sampled arm hands this module a ``tio.Queue``. The record was
    honest about it — ``dataset_chain: ["Queue"]`` plus a gap in ``incomplete``,
    per non-negotiable 14 — but honest-and-empty is exactly what let
    ``experiment_11_attention_none`` declare ``enable_log_scaling: true`` and run
    for 40 iterations against an uncompressed batch with nothing to compare.
    The declared half was right, the applied half was blank, and a blank half
    cannot contradict anything.
    """

    def test_the_walk_reaches_through_a_queue_to_its_subjects_dataset(self) -> None:
        dataset = _FakeQueue(_FakeDataset(_FakeCompose([_FakeTransform(p=0.3)])))
        record = build_snapshot_provenance(_config(), dataset=dataset)
        assert record["applied"]["dataset_chain"] == ["_FakeQueue", "_FakeDataset"]
        assert [t["name"] for t in record["applied"]["transforms"]] == ["_FakeTransform"]
        assert record["incomplete"] == []

    def test_a_privatized_transform_attribute_is_still_found(self) -> None:
        """torchio 1.2 renamed it; the record must not go blank on a rename."""
        dataset = _FakeQueue(_FakeTorchIODataset(_FakeCallableCompose([_FakeTransform(p=0.7)])))
        record = build_snapshot_provenance(_config(), dataset=dataset)
        assert record["applied"]["transforms"][0]["params"]["p"] == 0.7
        assert record["incomplete"] == []

    def test_a_non_callable_private_attribute_is_not_mistaken_for_a_transform(
        self,
    ) -> None:
        """``_transform`` is a private name; anything may live under it.

        Accepting it unconditionally would write a confident wrong
        ``{"name": "str"}`` into the artifact, which is worse than the gap it
        replaces — a reader can act on "unknown" but not on a plausible lie.
        Callability is the contract ``set_transform`` enforces, so it is the
        cheapest discriminator that admits every real Compose.
        """
        dataset = _FakeQueue(_FakeTorchIODataset("percentile"))
        record = build_snapshot_provenance(_config(), dataset=dataset)
        assert "transforms" not in record["applied"]
        assert any("_FakeTorchIODataset" in gap for gap in record["incomplete"])

    def test_the_gap_names_every_attribute_tried(self) -> None:
        """A gap naming one spelling sent this triage at the M4Raw loader.

        The loader was never at fault — the attribute had moved. "Exposes no
        .transform" reads as "this dataset has no transform" when the truth was
        "this reader looked in one place".
        """
        record = build_snapshot_provenance(_config(), dataset=_FakeDataset(None))
        gap = next(g for g in record["incomplete"] if "transform" in g)
        assert ".transform" in gap and "._transform" in gap

    def test_the_real_torchio_queue_is_walked_end_to_end(self) -> None:
        """The fakes above pin MY assumptions; this pins TorchIO's actual shape.

        A duck-type test cannot catch the next rename — it renames in lockstep
        with the source. This one fails the day ``subjects_dataset`` or
        ``_transform`` moves again, which is the failure class W3 exists to
        close, so it is worth the real-library import.
        """
        tio = pytest.importorskip("torchio")

        subject = tio.Subject(image=tio.ScalarImage(tensor=torch.rand(1, 8, 8, 4)))
        queue = tio.Queue(
            subjects_dataset=tio.SubjectsDataset(
                [subject], transform=tio.Compose([tio.ZNormalization()])
            ),
            max_length=2,
            samples_per_volume=1,
            sampler=tio.data.UniformSampler(patch_size=(4, 4, 2)),
            num_workers=0,
        )

        record = build_snapshot_provenance(_config(), dataset=queue)

        assert record["applied"]["dataset_chain"] == ["Queue", "SubjectsDataset"]
        assert [t["name"] for t in record["applied"]["transforms"]] == ["ZNormalization"]
        assert record["incomplete"] == [], (
            "the applied half went blank against real torchio — an attribute "
            "was renamed again; update _WRAPPER_ATTRS / _TRANSFORM_ATTRS"
        )


@pytest.mark.unit
class TestLinearization:
    def test_a_curve_module_is_reported_by_the_mode_it_was_built_with(self) -> None:
        """Read the mode off the CONSTRUCTED module, not re-derived from config.

        `HilbertOrder` and `ImageTopologyLinearizer` both permute the spatial
        axes before the sequence backbone sees them, so on a Mamba/SSM arm the
        tensor the network consumes is not the one the snapshot renders.
        """
        curve = torch.nn.Module()
        curve.mode = "hilbert"
        curve.shape = (64, 64)
        curve.register_buffer("permutation", torch.arange(4096))
        model = torch.nn.Sequential(curve, torch.nn.Conv2d(1, 1, 3))

        record = build_snapshot_provenance(_config(), model=model)
        assert record["model_input_linearization"] == [
            {"module": "Module", "mode": "hilbert", "shape": [64, 64]}
        ]

    def test_a_module_that_merely_has_a_mode_attribute_is_not_a_linearizer(self) -> None:
        impostor = torch.nn.Module()
        impostor.mode = "bilinear"  # e.g. an upsampler
        record = build_snapshot_provenance(_config(), model=impostor)
        assert record["model_input_linearization"] == []


@pytest.mark.unit
class TestNeverCorruptsTheArtifact:
    def test_a_mock_dataset_does_not_leak_its_repr_into_the_record(self) -> None:
        """A MagicMock auto-creates every attribute (#693).

        `tests/smoke/test_paradigm_step.py` drives this path with mocked envs;
        an unguarded getattr writes `<MagicMock id=...>` into snapshot.json.
        """
        record = build_snapshot_provenance(_config(), dataset=MagicMock(), model=MagicMock())
        # Recording nothing is fine; recording a transform NAMED "MagicMock" is
        # a confident wrong answer in the artifact whose job is to be trusted.
        assert "MagicMock" not in repr(record)
        assert any("test double" in gap for gap in record["incomplete"])

    def test_a_mock_config_yields_no_fabricated_declarations(self) -> None:
        record = build_snapshot_provenance(MagicMock())
        assert record["declared"].get("target_mode") is None

    @pytest.mark.parametrize("junk", [None, 0, "config", object()])
    def test_it_never_raises_whatever_it_is_handed(self, junk) -> None:
        """A diagnostic must never be the reason a training run dies."""
        record = build_snapshot_provenance(junk, dataset=junk, model=junk)
        assert record["source"] == "train"

    def test_a_wrapper_cycle_terminates_and_is_reported(self) -> None:
        outer = _FakeWrapper(None)
        middle = _FakeWrapper(outer)
        outer.inner = middle  # cycle
        record = build_snapshot_provenance(_config(), dataset=outer)
        assert any("deeper than" in gap for gap in record["incomplete"])

    def test_the_split_is_stamped_so_a_val_snapshot_cannot_claim_train(self) -> None:
        """Train and val are built by DIFFERENT Compose objects."""
        record = build_snapshot_provenance(_config(), source="val")
        assert record["source"] == "val"


# ── Run identity (#1299) ───────────────────────────────────────────────────
#
# A snapshot used to record no run identity at all, and the one artifact that
# carried it -- `provenance.json` -- is overwrite-on-launch. Two snapshots in
# one arm directory therefore could not be assumed to share a run, a config or a
# commit, and nothing in either file would reveal it.


@pytest.fixture(autouse=True)
def _clean_identity():
    """The identity is process-global, so it leaks between tests if left set."""
    from spectramr.infrastructure.training.snapshot_provenance import reset_run_identity

    reset_run_identity()
    yield
    reset_run_identity()


def _published(**over):
    from spectramr.infrastructure.training.snapshot_provenance import set_run_identity

    record = {
        "run_id": "exp11-20260821_101500-abc123def456",
        "run_name": "exp11",
        "started_at": "2026-08-21T10:15:00-05:00",
        "config_sha256": "0f1e2d3c",
        "git": {"sha": "abc123def456789", "branch": "dev", "dirty": True},
    }
    record.update(over)
    return set_run_identity(record)


def test_the_published_id_is_the_pipelines_own_not_a_new_one() -> None:
    """The whole point: one run, one id.

    A snapshot that minted its own would differ from `provenance.json`'s by its
    timestamp alone, and a reader correlating the two would conclude they came
    from different runs.
    """
    from spectramr.infrastructure.training.snapshot_provenance import run_identity

    _published()
    assert run_identity()["run_id"] == "exp11-20260821_101500-abc123def456"
    assert run_identity()["identity_source"] == "run_provenance"


def test_the_dirty_flag_travels_with_the_sha() -> None:
    """A clean sha does not reproduce a run whose tree was dirty, so the sha
    alone is not an identity."""
    from spectramr.infrastructure.training.snapshot_provenance import run_identity

    _published()
    record = run_identity()
    assert record["git_sha"] == "abc123def456789"
    assert record["git_dirty"] is True


def test_the_config_hash_is_carried_so_a_relaunch_is_detectable() -> None:
    """The `step_004000` case: the snapshot's config did not match the on-disk
    YAML, and there was no way to tell from the artifact. The hash is what makes
    that a comparison instead of an archaeology exercise."""
    from spectramr.infrastructure.training.snapshot_provenance import run_identity

    _published()
    assert run_identity()["config_sha256"] == "0f1e2d3c"


def test_two_runs_in_one_directory_are_distinguishable() -> None:
    """The failure #1299 reports, reduced: same arm dir, different launches."""
    from spectramr.infrastructure.training.snapshot_provenance import run_identity

    _published()
    first = run_identity()["run_id"]
    _published(run_id="exp11-20260821_113000-abc123def456")
    assert run_identity()["run_id"] != first


def test_nothing_published_still_yields_a_usable_id_that_admits_it() -> None:
    """A missing identity block is indistinguishable from a pre-#1299 snapshot,
    so there is always a record -- and it says when it is not the run's own."""
    from spectramr.infrastructure.training.snapshot_provenance import (
        IDENTITY_FALLBACK,
        run_identity,
    )

    record = run_identity()
    assert record["run_id"]
    assert record["identity_source"] == IDENTITY_FALLBACK


def test_the_fallback_is_stable_within_a_process() -> None:
    """Two snapshots from one un-published process must agree, or the artifact
    claims a relaunch happened between them."""
    from spectramr.infrastructure.training.snapshot_provenance import run_identity

    assert run_identity()["run_id"] == run_identity()["run_id"]


def test_publishing_after_a_fallback_was_minted_still_wins() -> None:
    """Provenance capture happens after the container is built, and a strategy
    constructed earlier may already have asked. The published record must not
    lose to a fallback that got there first."""
    from spectramr.infrastructure.training.snapshot_provenance import run_identity

    assert run_identity()["identity_source"] == "fallback"
    _published()
    assert run_identity()["identity_source"] == "run_provenance"


def test_an_empty_provenance_record_does_not_publish_a_hollow_identity() -> None:
    """`collect_run_provenance` is fail-open and legitimately returns `{}`. That
    must leave the fallback in charge, not stamp a row of Nones as canonical."""
    from spectramr.infrastructure.training.snapshot_provenance import (
        run_identity,
        set_run_identity,
    )

    assert set_run_identity({}) is None
    assert run_identity()["identity_source"] == "fallback"


def test_the_pipeline_actually_publishes_the_identity() -> None:
    """Wiring, by AST rather than by grep.

    Non-negotiable 8: the mechanism above is inert unless `run_training_pipeline`
    calls it. A call-site regex under-counts precisely where the code is tidiest,
    so the check walks the parsed module for the call by name.
    """
    import ast
    import inspect

    from spectramr.pipelines import train as train_mod

    tree = ast.parse(inspect.getsource(train_mod))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "set_run_identity" in called, (
        "the training pipeline never publishes the run identity, so every "
        "snapshot would fall back to a minted one"
    )


# --------------------------------------------------------------------------- #
# #1354 -- the dataset-wrapper walk stopped one link short
# --------------------------------------------------------------------------- #
def test_a_wrapper_delegating_through_base_is_traversed():
    """``base`` is a real delegate name in this tree, and it was not in the list.

    A wrapper the walk cannot step through truncates the applied-transform
    chain, which is the half of the provenance record the declared-vs-applied
    comparison is made against.
    """
    from spectramr.infrastructure.training.snapshot_provenance import _WRAPPER_ATTRS

    assert "base" in _WRAPPER_ATTRS


def test_the_walk_steps_through_a_base_attribute():
    from spectramr.infrastructure.training import snapshot_provenance

    class Leaf:
        transform = "LEAF-TRANSFORM"

        def __getitem__(self, i):
            return i

    class Wrapper:
        def __init__(self, inner):
            self.base = inner

        def __getitem__(self, i):
            return self.base[i]

    incomplete: list[str] = []
    chain, innermost = snapshot_provenance._dataset_chain(Wrapper(Leaf()), incomplete)
    assert "Leaf" in chain, chain
    assert isinstance(innermost, Leaf)


# ---------------------------------------------------------------------------
# timestep_floor: declared knob beside the floor the process resolved (#535)
# ---------------------------------------------------------------------------
#
# Non-negotiable 14's declared-vs-applied shape, for the diffusion timestep
# floor. Unlike a log line this survives `LoggingService.setup`'s level clamp
# and lands in an artifact, which is what lets someone reading a finished run
# answer "was the fully-sampled rung in this run?" without the config.


from spectramr.infrastructure.training import snapshot_provenance  # noqa: E402


class _FakeProcess:
    def __init__(self, floor: int) -> None:
        self._floor = floor

    def min_meaningful_timestep(self) -> int:
        return self._floor


class _FakeModel:
    def __init__(self, floor: int) -> None:
        self.kspace_process = _FakeProcess(floor)


def _accel_config(train_identity_rung: bool):
    return SimpleNamespace(
        data=None,
        undersampling=SimpleNamespace(train_identity_rung=train_identity_rung),
    )


def test_timestep_floor_records_declared_beside_applied():
    record = snapshot_provenance.build_snapshot_provenance(_accel_config(True), model=_FakeModel(0))
    assert record["timestep_floor"] == {
        "declared_train_identity_rung": True,
        "applied_min_timestep": 0,
    }


def test_timestep_floor_reports_the_divergence_rather_than_hiding_it():
    """A declared knob whose floor did NOT move is the finding, not an error.

    The applied half is read off the CONSTRUCTED process, never recomputed from
    config -- so a run where the two disagree renders both halves and lets the
    reader see it. Recomputing would make them agree by construction.
    """
    record = snapshot_provenance.build_snapshot_provenance(_accel_config(True), model=_FakeModel(1))
    assert record["timestep_floor"]["declared_train_identity_rung"] is True
    assert record["timestep_floor"]["applied_min_timestep"] == 1


def test_timestep_floor_is_absent_for_a_non_diffusion_arm():
    """Absent means "not applicable", and must not land in ``incomplete``.

    Most of the corpus has no ``kspace_process``; an incomplete entry per
    snapshot there would drown the entries that mean something failed.
    """
    record = snapshot_provenance.build_snapshot_provenance(
        _accel_config(False), model=SimpleNamespace()
    )
    assert "timestep_floor" not in record
    assert not any("floor" in entry for entry in record["incomplete"])


# ---------------------------------------------------------------------------
# The floor must survive the training wrappers (#1523)
# ---------------------------------------------------------------------------
#
# `build_snapshot_provenance` is handed `env.generator`
# (`strategies/base.py:1099`), which by snapshot time is WRAPPED: this cohort
# declares `parallel.strategy: deepspeed` and `ema.enabled: true`, and neither
# wrapper forwards `kspace_process`. A direct attribute read therefore returns
# None on the real launch path and the key vanishes from every snapshot -- while
# `test_timestep_floor_is_absent_for_a_non_diffusion_arm` stays green, because a
# wrapped diffusion arm and a non-diffusion arm are indistinguishable from
# outside. These plant the wrapper shapes so that blindness cannot come back.


class _Wrapper:
    """A DDP / DeepSpeed / ModelEma stand-in: wrapped module under ``.module``."""

    def __init__(self, inner: object) -> None:
        self.module = inner


class _CompiledWrapper:
    """``torch.compile``'s ``OptimizedModule``: inner module under ``_orig_mod``."""

    def __init__(self, inner: object) -> None:
        self._orig_mod = inner


def test_timestep_floor_is_found_through_a_ddp_wrapper():
    record = snapshot_provenance.build_snapshot_provenance(
        _accel_config(True), model=_Wrapper(_FakeModel(0))
    )
    assert record["timestep_floor"] == {
        "declared_train_identity_rung": True,
        "applied_min_timestep": 0,
    }


def test_timestep_floor_is_found_through_a_compile_wrapper():
    """An inline ``.module`` hop passes the DDP test above and fails this one.

    ``core.module_utils.unwrap_model`` is the elected SSOT precisely because the
    26 inline unwraps it replaced handled DP/DDP and missed ``_orig_mod``; a
    re-grown inline hop here would reproduce that gap silently.
    """
    record = snapshot_provenance.build_snapshot_provenance(
        _accel_config(True), model=_CompiledWrapper(_FakeModel(0))
    )
    assert record["timestep_floor"]["applied_min_timestep"] == 0


def test_timestep_floor_is_found_through_stacked_wrappers():
    """compile-then-DDP nests as ``DDP(OptimizedModule)``; both must peel."""
    record = snapshot_provenance.build_snapshot_provenance(
        _accel_config(True), model=_Wrapper(_CompiledWrapper(_FakeModel(1)))
    )
    assert record["timestep_floor"]["applied_min_timestep"] == 1


class _RaisingProcess:
    def min_meaningful_timestep(self) -> int:
        raise RuntimeError("accelerator not built")


def test_a_process_that_raises_is_reported_not_swallowed():
    """A failed probe names itself in ``incomplete`` (non-negotiable 14).

    Absence alone is ambiguous -- it is also what a non-diffusion arm produces.
    Without the entry, a floor that could not be read is indistinguishable from
    one that did not apply, which is the "silently omitted" shape the snapshot
    contract forbids. The run must not die for a provenance probe either, so the
    record is still returned.
    """
    model = SimpleNamespace(kspace_process=_RaisingProcess())
    record = snapshot_provenance.build_snapshot_provenance(_accel_config(True), model=model)
    assert "timestep_floor" not in record
    assert any("timestep floor probe failed" in entry for entry in record["incomplete"])
    assert any("accelerator not built" in entry for entry in record["incomplete"])


def test_declared_block_stamps_the_nex_fallback_policy() -> None:
    """``nex_fallback`` decides which reference a <3-rep contrast is graded
    against under leave-one-out, so the snapshot carries it beside
    ``nex_target_exclude_input`` (non-negotiable 14: declared knobs are stamped)."""
    record = build_snapshot_provenance(_config())
    assert record["declared"]["nex_fallback"] == "error"
    assert record["declared"]["nex_target_exclude_input"] is False


def test_declared_block_stamps_both_splits_targets_and_the_alpha() -> None:
    """Under r2r the two splits are built DIFFERENTLY, and the snapshot must say so.

    A snapshot naming only ``target_mode`` would describe the training half and
    silently misdescribe what the reported PSNR was graded against -- the exact
    divergence non-negotiable 14 exists to surface. ``r2r_alpha`` joins it
    because two runs differing only in alpha are different constructions that no
    other stamped field distinguishes.
    """
    config = _config()
    config.data.target_mode = "r2r"
    config.data.val_target_mode = "phase_aligned_mean"
    config.data.r2r_alpha = 0.25

    declared = build_snapshot_provenance(config)["declared"]

    assert declared["target_mode"] == "r2r"
    assert declared["val_target_mode"] == "phase_aligned_mean"
    # Anti-tautology: `model_dump`-style stamping emits every declared field, so
    # a presence assertion passes even if the stamp read `target_mode` twice.
    # Only the INEQUALITY can see that.
    assert declared["val_target_mode"] != declared["target_mode"]
    assert declared["r2r_alpha"] == 0.25


def test_declared_block_stamps_the_slice_level_index() -> None:
    """``slice_level_records`` decides whether a sample came from one record per
    slice (one slice read per repetition, a depth-1 subject) or one per group
    (#1757); the snapshot says which index built what it shows (non-negotiable
    14: declared knobs are stamped)."""
    assert build_snapshot_provenance(_config())["declared"]["slice_level_records"] is False
    on = _config()
    on.data.slice_level_records = True
    assert build_snapshot_provenance(on)["declared"]["slice_level_records"] is True
