"""Tests for the baseline adapter base contract.

Locks in ``TODO/backlog_baseline_replication_experiment_11.md`` Phase A.2.
The base class refuses to instantiate any concrete subclass that
leaves required provenance attributes at the sentinel value — this
is the CLAUDE.md pitfall #9 guard.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.baselines import BaselineAdapter, CoilHandling, FFTNorm
from spectramr.models.baselines._base import UpstreamLossFamily


class _CompleteAdapter(BaselineAdapter):
    """Minimal complete adapter — all required overrides set."""

    REPO_NAME = "fake_baseline"
    PAPER_REF = "arxiv:0000.00000"
    PREFERRED_MASK_TYPE = "equispaced"
    PREFERRED_FFT_NORM = FFTNorm.ORTHO
    COIL_HANDLING = CoilHandling.RSS
    UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L1

    def training_loss(self, x_0, batch=None):
        return x_0.abs().mean()

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return x


def test_complete_subclass_instantiates() -> None:
    """A subclass that overrides every required attribute instantiates cleanly."""
    adapter = _CompleteAdapter()
    assert adapter.REPO_NAME == "fake_baseline"
    assert adapter.PREFERRED_FFT_NORM is FFTNorm.ORTHO


def test_subclass_missing_repo_name_raises_at_class_creation() -> None:
    """Subclass that forgets ``REPO_NAME`` fails loudly at class-creation time."""
    with pytest.raises(TypeError, match="REPO_NAME"):

        class _Broken(BaselineAdapter):
            PAPER_REF = "arxiv:0000.0"
            PREFERRED_MASK_TYPE = "equispaced"
            UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L1

            def training_loss(self, x_0, batch=None):
                return x_0.abs().mean()

            def forward(self, x, *args, **kwargs):
                return x


def test_subclass_missing_paper_ref_raises() -> None:
    with pytest.raises(TypeError, match="PAPER_REF"):

        class _Broken(BaselineAdapter):
            REPO_NAME = "fake"
            PREFERRED_MASK_TYPE = "equispaced"
            UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L1

            def training_loss(self, x_0, batch=None):
                return x_0.abs().mean()

            def forward(self, x, *args, **kwargs):
                return x


def test_subclass_missing_mask_type_raises() -> None:
    with pytest.raises(TypeError, match="PREFERRED_MASK_TYPE"):

        class _Broken(BaselineAdapter):
            REPO_NAME = "fake"
            PAPER_REF = "arxiv:0000.0"

            def training_loss(self, x_0, batch=None):
                return x_0.abs().mean()

            def forward(self, x, *args, **kwargs):
                return x


def test_provenance_returns_declared_attrs() -> None:
    """``provenance()`` exposes the declared class attributes for the run summary."""
    adapter = _CompleteAdapter()
    prov = adapter.provenance()
    assert prov["repo_name"] == "fake_baseline"
    assert prov["paper_ref"] == "arxiv:0000.00000"
    assert prov["preferred_mask_type"] == "equispaced"
    assert prov["preferred_fft_norm"] == "ortho"
    assert prov["coil_handling"] == "rss"


def test_forward_is_abstract() -> None:
    """An adapter that only sets attributes (no ``forward``) is still abstract."""
    with pytest.raises(TypeError):

        class _NoForward(BaselineAdapter):
            REPO_NAME = "fake"
            PAPER_REF = "arxiv:0000.0"
            PREFERRED_MASK_TYPE = "equispaced"
            # no forward override

        _NoForward()


def test_base_class_itself_is_not_directly_instantiable() -> None:
    """``BaselineAdapter`` is abstract (it has @abstractmethod ``forward``)."""
    with pytest.raises(TypeError):
        BaselineAdapter()


def test_an_unresolvable_preferred_mask_type_is_refused() -> None:
    """The violation this check exists for, planted.

    Two adapters declared ``gaussian_density``, which names neither a registered
    accelerator nor a ``MaskType`` — so the attribute read as a checked
    convention while resolving to nothing (#2087).
    """
    with pytest.raises(TypeError, match="PREFERRED_MASK_TYPE") as excinfo:

        class _Unresolvable(BaselineAdapter):
            REPO_NAME = "x"
            PAPER_REF = "X2020:X"
            PREFERRED_MASK_TYPE = "gaussian_density"
            UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L1

            def training_loss(self, x_0, batch=None):
                return x_0.abs().mean()

            def forward(self, x, *args, **kwargs):
                return x

    assert "gaussian_density" in str(excinfo.value)


def test_both_vocabularies_are_accepted() -> None:
    """The accelerator registry and ``MaskType`` are each legal.

    FDB's peripheral-to-central mask exists only on the static ``MaskType`` path,
    so requiring an accelerator name would reject a correct declaration.
    """

    class _AcceleratorName(BaselineAdapter):
        REPO_NAME = "a"
        PAPER_REF = "A2020:A"
        PREFERRED_MASK_TYPE = "random_cartesian"
        UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L1

        def training_loss(self, x_0, batch=None):
            return x_0.abs().mean()

        def forward(self, x, *args, **kwargs):
            return x

    class _MaskTypeName(BaselineAdapter):
        REPO_NAME = "b"
        PAPER_REF = "B2020:B"
        PREFERRED_MASK_TYPE = "cartesian_peripheral"
        UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L1

        def training_loss(self, x_0, batch=None):
            return x_0.abs().mean()

        def forward(self, x, *args, **kwargs):
            return x

    assert _AcceleratorName.PREFERRED_MASK_TYPE == "random_cartesian"
    assert _MaskTypeName.PREFERRED_MASK_TYPE == "cartesian_peripheral"


def test_a_subclass_without_a_loss_family_is_refused() -> None:
    """PLANTED VIOLATION: the objective an upstream minimises must be declared.

    The strategy that runs these adapters checks an arm's declared ``losses:`` block
    against this attribute. Undeclared, the check silently passes anything, which is
    the unread-knob shape the attribute exists to close (non-negotiable 8).
    """
    with pytest.raises(TypeError, match="UPSTREAM_LOSS_FAMILY"):

        class _NoFamily(BaselineAdapter):
            REPO_NAME = "x"
            PAPER_REF = "y"
            PREFERRED_MASK_TYPE = "equispaced"
            PREFERRED_FFT_NORM = FFTNorm.ORTHO
            COIL_HANDLING = CoilHandling.RSS

            def training_loss(self, x_0, batch=None):
                return x_0.abs().mean()

            def forward(self, x, *args, **kwargs):
                return x


def test_a_loss_family_outside_the_vocabulary_is_refused() -> None:
    """PLANTED VIOLATION: a free-text family would never match an arm's declaration."""
    with pytest.raises(TypeError, match="UPSTREAM_LOSS_FAMILY"):

        class _BadFamily(BaselineAdapter):
            REPO_NAME = "x"
            PAPER_REF = "y"
            PREFERRED_MASK_TYPE = "equispaced"
            PREFERRED_FFT_NORM = FFTNorm.ORTHO
            COIL_HANDLING = CoilHandling.RSS
            UPSTREAM_LOSS_FAMILY = "charbonnier"

            def training_loss(self, x_0, batch=None):
                return x_0.abs().mean()

            def forward(self, x, *args, **kwargs):
                return x


# ── the declaration guard must know whether the class is abstract (#2157) ────
def test_an_abstract_intermediate_may_leave_the_sentinels() -> None:
    """PLANTED VIOLATION, in the direction that was broken.

    The guard documents an escape hatch for abstract intermediates, and it was
    unreachable: ``__init_subclass__`` runs inside ``type.__new__``, which
    ``ABCMeta.__new__`` calls *before* populating ``__abstractmethods__`` --
    and that name is a type slot, not an inherited attribute, so reading it
    there raises ``AttributeError`` and the ``getattr`` default answered "not
    abstract" for every class. The check therefore fired on exactly the classes
    it was written to skip, which is what stopped
    ``tests/unit/infrastructure/reporting/test_baseline_provenance.py`` from
    collecting -- and a collection error aborts the whole ``tests/unit/`` run.
    """

    class _Intermediate(BaselineAdapter):
        """Shares plumbing between adapters; declares nothing, implements nothing."""

    assert _Intermediate.__abstractmethods__, "the fixture must stay abstract"
    assert _Intermediate.REPO_NAME == "<MUST_OVERRIDE>"


def test_a_partially_abstract_subclass_is_still_skipped() -> None:
    """One abstract method left over is enough; the hatch is not all-or-nothing."""

    class _HalfDone(BaselineAdapter):
        def forward(self, x, *args, **kwargs):
            return x

    assert set(_HalfDone.__abstractmethods__) == {"training_loss"}


def test_completing_that_intermediate_makes_the_guard_bite() -> None:
    """Guards the two above from passing because the check was simply removed."""
    with pytest.raises(TypeError, match="REPO_NAME"):

        class _Concrete(BaselineAdapter):
            def training_loss(self, x_0, batch=None):
                return x_0.abs().mean()

            def forward(self, x, *args, **kwargs):
                return x


def test_the_check_runs_after_abstractness_is_known() -> None:
    """Pins the mechanism, not just the outcome.

    ``__init_subclass__`` cannot answer this question at all, so the validation
    has to hang off the metaclass. If someone moves it back, the two checks
    above go green for the wrong reason -- an abstract fixture would be
    rejected rather than skipped, and this asserts where the hook lives.
    """
    from abc import ABCMeta

    from spectramr.models.baselines import _base

    assert issubclass(type(BaselineAdapter), ABCMeta)
    assert type(BaselineAdapter) is _base._BaselineAdapterMeta
    assert not hasattr(BaselineAdapter, "__init_subclass_validates__")
    assert hasattr(BaselineAdapter, "_validate_concrete_declarations")
