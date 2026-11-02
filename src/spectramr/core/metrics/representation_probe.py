"""Linear-probe evaluation of a frozen SSL encoder over one validation pass.

The standard reading of a self-supervised representation (I-JEPA, Assran et al. 2023) is a
linear classifier fitted on frozen embeddings. Here the label is the acquisition contrast a
validation batch carries (``contrast_idx``) and the split is subject-disjoint when subject
ids are present, as in :func:`~spectramr.core.metrics.disentanglement.linear_probe`, which
this accumulator feeds. Like the disentanglement readings it is not a registry metric: the
registry contract is ``(pred, target)`` images, and a probe needs the whole pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from spectramr.core.metrics.disentanglement import linear_probe

#: The keys :meth:`RepresentationProbeAccumulator.summary` writes.
PROBE_METRIC_KEYS = frozenset(
    {
        "val_probe_contrast_acc",
        "val_probe_contrast_chance",
        "val_probe_contrast_above_chance",
        "val_probe_split_by_subject",
    }
)


class RepresentationProbeAccumulator:
    """Rows of ``(embedding, contrast label, subject id)`` gathered across a validation pass."""

    def __init__(self, ridge: float = 1.0) -> None:
        self.ridge = ridge
        self._features: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._groups: list[str] = []
        self._batches_with_groups = 0

    @property
    def n_rows(self) -> int:
        return sum(int(t.shape[0]) for t in self._labels)

    def add(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        groups: Sequence[str] | None = None,
    ) -> None:
        """Append one batch: ``features`` ``(N, D)``, ``labels`` ``(N,)``, optional subject ids.

        Rows stay on their device until :meth:`summary`, which moves them once.

        Raises:
            ValueError: on a row-count mismatch, or when subject ids accompany some batches
                of a pass and not others (the split would silently change meaning).
        """
        labels = labels.detach().reshape(-1)
        if features.dim() != 2 or features.shape[0] != labels.numel():
            raise ValueError(
                f"probe rows: features {tuple(features.shape)} for {labels.numel()} labels"
            )
        if groups is not None and len(groups) != labels.numel():
            raise ValueError(f"{len(groups)} subject ids for {labels.numel()} rows")
        if self._labels and (groups is None) != (self._batches_with_groups == 0):
            raise ValueError("subject ids must accompany every batch of a pass, or none")
        self._features.append(features.detach())
        self._labels.append(labels)
        if groups is not None:
            self._groups.extend(str(g) for g in groups)
            self._batches_with_groups += 1

    def summary(self) -> dict[str, float]:
        """Held-out probe accuracy, its chance level and margin, and the split kind.

        Raises:
            ValueError: from :func:`linear_probe` when the pass cannot support a held-out
                probe (fewer than 4 rows, 2 contrasts or 2 subjects).
        """
        features = torch.cat(self._features).cpu()
        labels = torch.cat(self._labels).cpu().long()
        groups = self._groups or None
        acc, chance = linear_probe(features, labels, ridge=self.ridge, groups=groups)
        return {
            "val_probe_contrast_acc": acc,
            "val_probe_contrast_chance": chance,
            "val_probe_contrast_above_chance": acc - chance,
            # 1 = subject-disjoint split, 0 = row halves.
            "val_probe_split_by_subject": float(groups is not None),
        }


__all__ = ["PROBE_METRIC_KEYS", "RepresentationProbeAccumulator"]
