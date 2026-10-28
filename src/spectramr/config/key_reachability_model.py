"""Verdict types and the internal index records for :mod:`key_reachability`.

Split out of that module in the Wave 0 exit-criterion work (#1400): it was 814
LOC against the 300 ceiling (NN20). This is the leaf of the dependency chain
``model <- collect <- index <- key_reachability`` -- it imports no sibling, so
the analysis modules can all depend on one definition of each record (NN17).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

_LINE_STRIDE = 1_000_000


class ReadEvidence(StrEnum):
    """What the analysis actually found, which ``reachable`` alone cannot say.

    ``reachable=False`` has two causes that a wire-or-delete decision must not
    confuse, and reading them apart must not require string-matching ``reason``:

    ``NO_LIVE_READ``
        Reads exist and every one was positively shown unable to run. This is a
        finding about the *call graph* and it is as strong as this analysis gets.

    ``NO_READ_FOUND``
        No token naming the key anywhere in the analysed read zone. Usually that
        means nothing reads it -- but it is also what a consumer that names no
        token looks like: ``**cfg.model_dump()`` splatting, ``**kwargs``
        forwarding, or a field name built at runtime
        (``getattr(cfg, f"compute_{name}")``, where :func:`_name_tokens` can only
        see the fragment ``"compute_"``). **A human must read the would-be
        consumer before acting on one of these.**

    ``ACCESSOR_READ``
        A human did read that consumer, and the consumer now declares what it
        touches. The key was not found by token, but an accessor whose read set
        is exported beside it claims this exact path -- so the key is read, and
        this analysis simply cannot see the read. Distinct from ``LIVE_READ``
        because the evidence is a *declaration*, not a call graph: it is only as
        good as the export, which is why the exporting module owes a test that
        the declaration matches what the accessor executes.
    """

    LIVE_READ = "live_read"
    ACCESSOR_READ = "accessor_read"
    NO_LIVE_READ = "no_live_read"
    NO_READ_FOUND = "no_read_found"


@dataclass(frozen=True)
class ReachabilityVerdict:
    """Whether a config key has a read that can run, and the reads it judged.

    Attributes:
        reachable: ``False`` only when no read was shown able to run. Do **not**
            read this as "dead" on its own -- pair it with ``evidence``, which
            says whether that is a call-graph finding or an absence of evidence.
        sites: ``file:line`` of every read of the key's leaf name, live or not.
            Part of the contract: a wire-or-delete decision needs the reads.
        reason: Why. Names the ambiguity when the verdict is reachable-by-doubt,
            and names the dead scope when it is not. Prose, for humans; branch on
            ``evidence`` instead.
        evidence: The machine-readable discriminator. See :class:`ReadEvidence`.
    """

    reachable: bool
    sites: tuple[str, ...]
    reason: str
    evidence: ReadEvidence


def accessor_verdict(
    dotted_path: str,
    leaf: str,
    sites: tuple[str, ...],
    accessor: str,
) -> ReachabilityVerdict:
    """The verdict for a path an accessor declares it reads.

    ``sites`` stays the AST sites -- usually empty, because a token-less read is
    why the path is in the map at all. Fabricating a site for a token that does
    not exist would make ``sites`` a lie, and the ledger tests that ask "does this
    entry really have read sites" would read the fabrication as evidence. The
    accessor belongs in ``reason``, where prose goes.
    """
    return ReachabilityVerdict(
        reachable=True,
        sites=sites,
        reason=(
            f"no live token read of `{leaf}`, but an accessor declares it reads "
            f"`{dotted_path}`: {accessor}. The read names no token this index can "
            "see, so the declaration is the evidence -- it is checked against what "
            "the accessor executes by the exporting module's own tests."
        ),
        evidence=ReadEvidence.ACCESSOR_READ,
    )


@dataclass(frozen=True)
class ClassVerdict:
    """Whether a class can be instantiated on a path that runs."""

    live: bool
    reason: str
    evidence: tuple[str, ...]


@dataclass
class _Scope:
    """One executable region: a module body, a function or a method body."""

    file: str
    kind: str  # "module" | "function" | "method"
    qualname: str
    lineno: int
    parent: int | None
    class_name: str | None
    #: True for a scope a config read may be counted from: inside the package
    #: and outside ``config/schemas/``.
    read_zone: bool
    decorated: bool = False
    mentions: set[str] | None = None
    constructs: set[str] | None = None


@dataclass
class _Class:
    """A ``class`` statement, wherever it was found."""

    name: str
    file: str
    lineno: int
    bases: tuple[str, ...]
    decorated: bool
    method_scopes: list[int]


@dataclass
class _Index:
    scopes: list[_Scope]
    classes: dict[str, list[_Class]]
    #: leaf name -> packed ``scope << stride | lineno`` for every textual mention
    sites: dict[str, list[int]]
    #: name -> packed sites where it appears in call position
    constructions: dict[str, list[int]]
    funcs_by_name: dict[str, list[int]]
    live_scopes: set[int]
    live_classes: dict[str, str]
