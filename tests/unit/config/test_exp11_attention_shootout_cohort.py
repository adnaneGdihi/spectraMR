"""The attention shootout's cohort-level invariants.

Twelve arms whose contract is that they differ in ``attention_type`` (or a
knob inside it) and nothing else that could move a metric. Two invariants, and
they are invariants of different kinds -- one scientific, one declarative.

**Training exposure** is the scientific one, and it has two factors.
``max_iterations`` counts micro-batches -- one ``next(data_iter)`` per
iteration -- so the samples an arm sees are ``data.loader.batch_size x
max_iterations``, and the optimizer and scheduler step once every
``accumulation_steps`` iterations. Equal samples together with equal effective
batch (``batch_size x accumulation_steps``) therefore pin equal optimizer steps,
which is why there is no third test for them. Four arms ran ``1 x 4`` at the
control's 70000 iterations, holding the effective batch while seeing half the
samples and taking half the steps (#1265); the owner's 2026-10-04 decision is
one batch size and one accumulation on every arm, which the micro-batch test
pins, and a samples-seen budget that survives any memory fallback.

Rank count is the other multiplier and no YAML can pin it: ``num_devices`` is
overwritten from ``LOCAL_WORLD_SIZE`` by ``pipelines/distributed.py`` on any
``train-distributed`` launch. The cohort is therefore comparable at any world
size PROVIDED every arm is launched at the same one. That is launch discipline,
not config, and it is stated here because this file is where someone will look.

**First-class scientific metadata** is the declarative one. ``hypothesis`` /
``primary_metric`` / ``baseline`` began life smuggled into ``metadata.tags``
because ``ExperimentMetadataSchema`` was ``extra="forbid"`` and had nowhere else
to put them. The 2026-06 validation campaign made them real fields; six arms
moved and four did not, which is not cosmetic: ``check_scientific_metadata``
returns ``[]`` outright when ``metadata.primary_metric`` is unset, so those four
were skipping the check entirely while looking, in the YAML, exactly as
annotated as the rest. A free-form bag that nothing reads is the facade shape --
the declaration was present and inert.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from spectramr.config.settings import TrainingSettings  # noqa: E402
from tests.utils.corpus import tracked_yamls  # noqa: E402

COHORT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "inprogress"
    / "kspace_filling"
    / "attention_shootout"
)
CONTROL = "experiment_11_attention_none"
#: Arms whose micro-batch is lifted for memory (job 8818221 OOM at batch 4). The samples and
#: effective-batch tests still bind them, so they train the same 420000 samples in 105000 steps.
MEMORY_FALLBACK = frozenset({"experiment_11_attention_kan_dual_domain_sparse"})


def _arms() -> dict[str, TrainingSettings]:
    """``stem -> resolved settings`` for every shootout arm.

    Resolved, never raw YAML: this cohort has already been bitten once by a
    raw-text predicate that went vacuously green when a canonical-key drain
    moved a block without changing the resolved document
    (``test_exp11_attention_none_lookahead``). A ``metadata.tags.primary_metric``
    vs ``metadata.primary_metric`` split is exactly that shape of difference.
    """
    return {
        p.stem: TrainingSettings.from_yaml(str(p))
        for p in tracked_yamls(COHORT, "experiment_11_attention_*.yaml", recursive=False)
    }


@pytest.fixture(scope="module")
def arms() -> dict[str, TrainingSettings]:
    return _arms()


def _meta(cfg: TrainingSettings, field: str):
    """Read ``metadata.<field>`` through either shape it resolves to.

    ``TrainingSettings.metadata`` is ``dict[str, Any] | None``
    (``config/settings.py:176``), while the sub-schema path uses
    ``ExperimentMetadataSchema`` (an object). ``check_scientific_metadata``
    carries the same two-branch read for the same reason; a test that assumed
    only the object shape would ``AttributeError`` on every arm rather than
    assert anything -- loud, but about the wrong thing.
    """
    meta = getattr(cfg, "metadata", None)
    if isinstance(meta, dict):
        return meta.get(field)
    return getattr(meta, field, None) if meta is not None else None


def test_the_cohort_is_present(arms) -> None:
    """Anti-vacuity for every assertion below: an empty or singleton mapping
    would satisfy the `all(...)` shapes trivially."""
    assert len(arms) >= 2, f"cohort has {len(arms)} arm(s); comparisons are vacuous"
    assert CONTROL in arms, "the control arm is missing from the cohort"


def _exposure(cfg: TrainingSettings) -> tuple[int, int, int]:
    """``(micro-batch, accumulation, max_iterations)`` as the run reads them.

    ``data.loader.batch_size`` is the one the DataLoader is built from;
    ``training.batch_size`` has no reader in ``src/`` and must not stand in.
    """
    return (
        cfg.data.loader.batch_size,
        cfg.optimization.gradient.accumulation_steps,
        cfg.training.max_iterations,
    )


def _divergent(
    exposure: dict[str, tuple[int, int, int]], quantity, exempt: frozenset[str] = frozenset()
) -> dict[str, object]:
    """Arms whose ``quantity(batch, accum, iters)`` differs from the control's."""
    control = quantity(*exposure[CONTROL])
    return {
        s: v for s, e in exposure.items() if s not in exempt and (v := quantity(*e)) != control
    }


def _samples(batch: int, _accum: int, iters: int) -> int:
    return batch * iters


def _effective_batch(batch: int, accum: int, _iters: int) -> int:
    return batch * accum


def _micro_batch(batch: int, accum: int, _iters: int) -> tuple[int, int]:
    return (batch, accum)


@pytest.mark.parametrize(
    ("quantity", "what"),
    [
        (_samples, "samples seen (data.loader.batch_size x max_iterations)"),
        (_effective_batch, "effective batch (batch_size x accumulation_steps)"),
        (_micro_batch, "(batch_size, accumulation_steps)"),
    ],
    ids=["samples", "effective-batch", "micro-batch"],
)
def test_every_arm_matches_the_control_exposure(arms, quantity, what: str) -> None:
    """Samples and effective batch are the science; the micro-batch is the
    owner's 2026-10-04 ruling on #1265 and the one a memory fallback may lift,
    provided the samples test stays green."""
    exempt = MEMORY_FALLBACK if quantity is _micro_batch else frozenset()
    divergent = _divergent({s: _exposure(c) for s, c in arms.items()}, quantity, exempt)
    assert not divergent, (
        f"{what} differs from the control on {divergent}. That is a second axis "
        "in a cohort whose contract is attention_type alone (pitfall #17): "
        "max_iterations counts micro-batches, so a smaller batch at the same "
        "budget sees fewer samples and takes fewer optimizer steps."
    )


_CONTROL_EXPOSURE = (2, 2, 70000)


@pytest.mark.parametrize(
    ("planted", "red"),
    [
        ((1, 4, 70000), True),  # #1265's shape: effective batch held, samples halved
        ((2, 2, 35000), True),  # budget cut alone
        ((1, 4, 140000), False),  # the sample-equal memory fallback
    ],
    ids=["halved-batch-same-budget", "budget-alone", "sample-equal-fallback"],
)
def test_the_samples_rule_turns_red_on_a_planted_arm(planted, red: bool) -> None:
    exposure = {CONTROL: _CONTROL_EXPOSURE, "planted": planted}
    assert bool(_divergent(exposure, _samples)) is red


def test_the_memory_exemption_lifts_only_the_micro_batch() -> None:
    """PLANTED: an exempt arm may split 2 x 2, an unlisted one may not, and the exemption
    never reaches the samples or effective-batch rules."""
    split = {CONTROL: (4, 1, 105000), "x": (2, 2, 210000)}
    assert _divergent(split, _micro_batch) == {"x": (2, 2)}
    assert _divergent(split, _micro_batch, frozenset({"x"})) == {}
    assert _divergent(split, _samples, frozenset({"x"})) == {}
    short = {CONTROL: (4, 1, 105000), "x": (2, 2, 105000)}
    assert _divergent(short, _samples) == {"x": 210000}
    assert set(_arms()) >= MEMORY_FALLBACK


def test_the_rule_reads_the_loader_batch_not_training_batch(tmp_path: Path) -> None:
    """Planted on a resolved YAML: only ``data.loader.batch_size`` moves, so a
    predicate reading the inert ``training.batch_size`` would stay green."""
    import yaml

    doc = yaml.safe_load((COHORT / f"{CONTROL}.yaml").read_text())
    doc["data"]["loader"]["batch_size"] = 1
    planted = tmp_path / "planted.yaml"
    planted.write_text(yaml.safe_dump(doc))
    control = _exposure(TrainingSettings.from_yaml(str(COHORT / f"{CONTROL}.yaml")))
    exposure = {CONTROL: control, "planted": _exposure(TrainingSettings.from_yaml(str(planted)))}
    assert _divergent(exposure, _samples) == {"planted": control[2]}


def test_the_axis_under_test_actually_varies(arms) -> None:
    """Anti-vacuity in the other direction: a cohort that agreed on
    ``attention_type`` too would pass every assertion here and be no
    experiment at all."""
    kinds = {stem: cfg.model.model_kwargs.get("attention_type") for stem, cfg in arms.items()}
    assert len(set(kinds.values())) > 1, f"all arms share one attention_type: {kinds}"
    assert kinds[CONTROL] == "none", (
        f"the control declares attention_type={kinds[CONTROL]!r}, not 'none'"
    )


@pytest.mark.parametrize("field", ["hypothesis", "primary_metric"])
def test_every_arm_declares_its_science_first_class(arms, field: str) -> None:
    """``metadata.tags`` is a free-form bag; the audit reads the real fields.

    ``check_scientific_metadata`` returns ``[]`` when ``primary_metric`` is
    unset, so an arm carrying it under ``tags`` is not annotated-but-messy, it
    is unchecked -- and indistinguishable from an annotated arm by eye.
    """
    missing = [stem for stem, cfg in arms.items() if not _meta(cfg, field)]
    assert not missing, (
        f"metadata.{field} is unset on {missing}. If the value is under "
        "`metadata.tags:`, move it up -- tags is free-form and nothing reads it, "
        "so the arm silently skips check_scientific_metadata."
    )


def test_every_non_control_arm_names_its_baseline(arms) -> None:
    """The control needs none -- it IS the baseline (``metadata.role``)."""
    missing = [stem for stem, cfg in arms.items() if stem != CONTROL and not _meta(cfg, "baseline")]
    assert not missing, f"metadata.baseline is unset on {missing}"


def test_a_named_baseline_exists_in_the_cohort(arms) -> None:
    """A baseline pointing at nothing is worse than none: it reads as a
    comparison that was made."""
    dangling = {
        stem: _meta(cfg, "baseline")
        for stem, cfg in arms.items()
        if _meta(cfg, "baseline") and _meta(cfg, "baseline") not in arms
    }
    assert not dangling, f"metadata.baseline names a non-existent arm: {dangling}"


def test_the_science_fields_are_not_also_left_in_tags(arms) -> None:
    """Move, not copy. Two homes for one value is how the two drift apart, and
    the tags copy is the one that would go stale unnoticed."""
    duplicated = {
        stem: [f for f in ("hypothesis", "primary_metric", "baseline") if f in tags]
        for stem, cfg in arms.items()
        if isinstance(tags := (_meta(cfg, "tags") or {}), dict)
        and any(f in tags for f in ("hypothesis", "primary_metric", "baseline"))
    }
    assert not duplicated, (
        f"scientific metadata is still under metadata.tags on {duplicated}; it "
        "belongs in the first-class fields alone (config/schemas/base.py)."
    )
