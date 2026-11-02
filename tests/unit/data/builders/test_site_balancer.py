"""Tests for the domain/site batch balancers.

Review finding (2026-07-01): ``DomainConfigSchema.weight`` was validated,
stamped, and threaded into ``build_balancer`` — but no balancer consumed it.
``StratifiedBalancer`` stored ``self.weights`` and never read it; the default
``RoundRobinBalancer`` did not even accept it. A domain-adaptation config asking
for 3x oversampling of the target domain silently ran 1:1 (pitfall #15).

Fix: ``RoundRobinBalancer`` (one domain per step) honours weights via an
integer visit schedule; ``StratifiedBalancer`` (one batch per domain per step)
cannot express differential weights and now raises rather than no-op.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from spectramr.data.builders.site_balancer import (
    RoundRobinBalancer,
    StratifiedBalancer,
    build_balancer,
)


class _FakeLoader:
    """Minimal DataLoader stand-in: yields ``n`` unique-labelled batches."""

    def __init__(self, name: str, n: int) -> None:
        self._name = name
        self._n = n

    def __iter__(self):
        return iter([f"{self._name}:{i}" for i in range(self._n)])

    def __len__(self) -> int:
        return self._n


def _domain_counts(balancer) -> Counter:
    return Counter(step["domain"] for step in balancer)


def test_round_robin_uniform_weights_is_balanced() -> None:
    """Unit weights reproduce strict 1:1 round-robin (backward compatible)."""
    loaders = {"source": _FakeLoader("source", 4), "target": _FakeLoader("target", 4)}
    counts = _domain_counts(RoundRobinBalancer(loaders))
    assert counts["source"] == counts["target"]


def test_round_robin_honours_weighted_oversampling() -> None:
    """weight 3.0 on 'target' ⇒ ~3x as many target steps as source."""
    loaders = {"source": _FakeLoader("source", 4), "target": _FakeLoader("target", 4)}
    balancer = RoundRobinBalancer(loaders, weights={"source": 1.0, "target": 3.0})
    counts = _domain_counts(balancer)
    assert counts["target"] == 3 * counts["source"], counts
    assert len(balancer) == sum(counts.values())


def test_round_robin_schedule_is_interleaved_not_blocked() -> None:
    """The oversampled domain is spread out, not clustered as [s, t, t, t]."""
    loaders = {"source": _FakeLoader("source", 2), "target": _FakeLoader("target", 2)}
    balancer = RoundRobinBalancer(loaders, weights={"source": 1.0, "target": 3.0})
    # One schedule cycle interleaves as [source, target, target, target].
    assert balancer._schedule == ["source", "target", "target", "target"]


def test_build_balancer_round_robin_threads_weights() -> None:
    """The factory (default strategy) must forward weights to round-robin."""
    loaders = {"source": _FakeLoader("source", 3), "target": _FakeLoader("target", 3)}
    balancer = build_balancer(
        loaders, strategy="round_robin", weights={"source": 1.0, "target": 2.0}
    )
    counts = _domain_counts(balancer)
    assert counts["target"] == 2 * counts["source"]


def test_stratified_rejects_non_uniform_weights() -> None:
    """Stratified cannot express differential weights — it must raise, not no-op."""
    loaders = {"source": _FakeLoader("source", 3), "target": _FakeLoader("target", 3)}
    with pytest.raises(NotImplementedError, match="differentially oversample"):
        StratifiedBalancer(loaders, weights={"source": 1.0, "target": 3.0})


def test_stratified_uniform_weights_ok() -> None:
    """Uniform weights are a well-defined no-op and must still construct."""
    loaders = {"source": _FakeLoader("source", 3), "target": _FakeLoader("target", 3)}
    balancer = StratifiedBalancer(loaders, weights={"source": 2.0, "target": 2.0})
    steps = list(balancer)
    assert len(steps) == 3
    assert all(set(step) == {"source", "target"} for step in steps)


# ── WS8: import-surface guard after the unused-typing.Iterable cleanup ─────────


def test_module_imports_and_exposes_balancers():
    """Guards the WS8 import cleanup (dropped unused ``typing.Iterable``): the
    module must still import and expose its balancer surface."""
    import spectramr.data.builders.site_balancer as sb

    assert hasattr(sb, "build_balancer")
    assert hasattr(sb, "RoundRobinBalancer")
    assert hasattr(sb, "Balancer")


# ── DomainLabelledBalancer: one flat batch + per-sample domain_label ──────────


class _DictLoader:
    """Yields ``n`` collated dict batches of ``bs`` samples filled with ``value``."""

    def __init__(self, value: float, n: int, bs: int = 2, extra: str | None = None) -> None:
        self._value, self._n, self._bs, self._extra = value, n, bs, extra

    def __iter__(self):
        for i in range(self._n):
            batch = {
                "input": torch.full((self._bs, 1, 4, 4), self._value),
                "meta": {"step": torch.full((self._bs,), float(i)), "path": ["p"] * self._bs},
            }
            if self._extra:
                batch[self._extra] = torch.zeros(self._bs)
            yield batch

    def __len__(self) -> int:
        return self._n


def test_domain_labelled_concatenates_and_labels_per_sample() -> None:
    balancer = build_balancer(
        {"synthetic": _DictLoader(1.0, 2, bs=2), "real": _DictLoader(-1.0, 2, bs=3)},
        strategy="domain_labelled",
    )
    batch = next(iter(balancer))
    assert batch["input"].shape == (5, 1, 4, 4)
    assert batch["domain_label"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    # Labels line up with the domain each sample came from.
    assert batch["input"][:2].eq(1.0).all() and batch["input"][2:].eq(-1.0).all()
    assert batch["meta"]["step"].shape == (5,) and len(batch["meta"]["path"]) == 5


def test_domain_labelled_epoch_follows_the_source_and_cycles_the_target() -> None:
    balancer = build_balancer(
        {"synthetic": _DictLoader(1.0, 3), "real": _DictLoader(-1.0, 1)},
        strategy="domain_labelled",
    )
    assert len(balancer) == 3
    assert len(list(balancer)) == 3


def test_domain_labelled_refuses_mismatched_keys() -> None:
    balancer = build_balancer(
        {"synthetic": _DictLoader(1.0, 1, extra="marker"), "real": _DictLoader(-1.0, 1)},
        strategy="domain_labelled",
    )
    with pytest.raises(ValueError, match="disagree"):
        next(iter(balancer))


def test_domain_labelled_needs_exactly_two_uniformly_weighted_domains() -> None:
    three = {k: _DictLoader(0.0, 1) for k in ("a", "b", "c")}
    with pytest.raises(ValueError, match="exactly 2"):
        build_balancer(three, strategy="domain_labelled")
    two = {k: _DictLoader(0.0, 1) for k in ("a", "b")}
    with pytest.raises(NotImplementedError, match="non-uniform"):
        build_balancer(two, strategy="domain_labelled", weights={"a": 1.0, "b": 3.0})


def test_schema_rejects_domain_labelled_with_three_domains() -> None:
    from spectramr.config.schemas.data import MultiDomainConfigSchema

    domains = [{"name": n} for n in ("a", "b", "c")]
    with pytest.raises(ValueError, match="exactly 2"):
        MultiDomainConfigSchema(enabled=True, balancing="domain_labelled", domains=domains)
    ok = MultiDomainConfigSchema(enabled=True, balancing="domain_labelled", domains=domains[:2])
    assert ok.balancing == "domain_labelled"
