"""Who a M4Raw file belongs to, parsed once.

M4Raw ships ``<patient>_<contrast><NN>.h5``. Three different questions are
answered by three different slices of that name, and conflating them is the
mistake this module exists to stop:

``subject``
    The patient. The **independence** unit: two contrasts of one patient share
    an anatomy, so counting them as separate draws inflates ``n`` in a
    Hoeffding bound (#1707).

``repetition_group``
    ``(patient, contrast)`` -- the set of repeated acquisitions the NEX target
    averages. This is what ``M4RawRepetitionDataset`` groups on, and it is NOT
    an independence unit.

``repetition``
    Which acquisition of that group. Repetitions are **exchangeable by
    construction**: the same anatomy measured again, differing only in an
    independent noise draw.

The grouping rule used to be written twice -- ``stem[:-2]`` inside the dataset
and ``file_id[:-2]`` in the manifest generator -- so this is the one owner both
now call. A third copy was about to be added for the batch identity fields,
which is what prompted electing one.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["M4RawIdentity", "parse_m4raw_file_id", "repetition_group_key"]

#: The trailing repetition counter, e.g. ``01`` in ``2022061001_T101``.
_REPETITION_DIGITS = 2


class M4RawIdentity(NamedTuple):
    """The three identities latent in one M4Raw file name."""

    subject: str
    contrast: str
    repetition: str
    repetition_group: str


def repetition_group_key(file_id: str) -> str:
    """The NEX group a file belongs to: ``(patient, contrast)``.

    Short names cannot carry a two-digit suffix, so they are their own group
    rather than being truncated into a shared one.
    """
    if len(file_id) < _REPETITION_DIGITS + 1:
        return file_id
    return file_id[:-_REPETITION_DIGITS]


def parse_m4raw_file_id(file_id: str) -> M4RawIdentity:
    """Decompose ``<patient>_<contrast><NN>`` without assuming it parses.

    A name that does not carry the convention degrades to ``subject == file_id``
    with an empty contrast and repetition. That is deliberate: the identity is
    used to GROUP, and putting an unparsed file in a group of its own is
    correct, whereas guessing a subject would silently merge two patients.

    Examples:
        >>> parse_m4raw_file_id("2022061001_T101")
        M4RawIdentity(subject='2022061001', contrast='T1', repetition='01', ...)
        >>> parse_m4raw_file_id("2022061001_FLAIR02").contrast
        'FLAIR'
    """
    group = repetition_group_key(file_id)
    repetition = file_id[-_REPETITION_DIGITS:] if group != file_id else ""
    if not repetition.isdigit():
        # Not the convention: one group of its own, no contrast claimed.
        return M4RawIdentity(subject=file_id, contrast="", repetition="", repetition_group=file_id)
    subject, _, contrast = group.rpartition("_")
    if not subject:
        # No separator, so the whole stem names the subject and no contrast is
        # claimed -- reading the leading characters as one would be a guess.
        return M4RawIdentity(
            subject=group, contrast="", repetition=repetition, repetition_group=group
        )
    return M4RawIdentity(
        subject=subject, contrast=contrast, repetition=repetition, repetition_group=group
    )
